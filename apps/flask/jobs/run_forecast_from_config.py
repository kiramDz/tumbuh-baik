from pymongo import MongoClient
from datetime import datetime
import time
import pandas as pd
import traceback
import os
from flask import jsonify
from dotenv import load_dotenv
from holt_winter.hw_dynamic_2 import run_optimized_hw_analysis


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("No MONGODB_URI set in environment variables!")

client = MongoClient(MONGO_URI)
db = client["tugas_akhir"]

def convert_objectid(obj):
    """Convert ObjectId to string for JSON serialization"""
    if isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_objectid(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return convert_objectid(obj.__dict__)
    else:
        return obj

def is_valid_column(collection_name, column_name, client):
    """Check if column is suitable for forecasting (not ID, date, etc.)"""
    forbidden_keywords = ["id", "date", "time", "timestamp", "_id"]
    if any(keyword in column_name.lower() for keyword in forbidden_keywords):
        return False, f"Column {column_name} contains forbidden keyword for forecasting"
    
    # Optional: Check if column contains numeric data
    try:
        sample_doc = client["tugas_akhir"][collection_name].find_one({column_name: {"$exists": True}})
        if sample_doc and not isinstance(sample_doc[column_name], (int, float)):
            return False, f"Column {column_name} contains non-numeric data"
        return True, None
    except Exception as e:
        return False, f"Error checking column {column_name}: {str(e)}"

def run_forecast_from_config():
    try:

        # Kosongkan collection temp-hw di awal
        collections_to_clean = ["temp-hw", "temp-decompose", "decompose"]
        for coll_name in collections_to_clean:
                db[coll_name].delete_many({})

        config = db.forecast_configs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running"}},
            return_document=True
        )
        
        if not config:
            return jsonify({"message": "No pending forecast config found."}), 200
        
        # Kosongkan collection holt-winter di awal (tanpa perlu pengecekan)
        db["holt-winter"].delete_many({})
        
        # Ambil info kolom yang akan dianalisis
        name = config.get("name", f"forecast_{int(time.time())}")
        columns = config.get("columns", [])
        forecast_coll = config.get("forecastResultCollection")
        config_id = str(config["_id"])

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            is_valid, error_msg = is_valid_column(collection, column, client)
            if not is_valid:
                db.forecast_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "errorMessage": error_msg}}
                )
                return jsonify({"error": error_msg}), 400
        
        results = []
        forecast_data = {}  # Untuk menyimpan semua forecast berdasarkan tanggal
        decompose_data = {}
        error_metrics_list = []

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            
            print(f"[INFO] Processing {collection} - {column}")
            
            try:
                # Jalankan analisis Holt-Winter
                result = run_optimized_hw_analysis(
                    collection_name=collection,
                    target_column=column,
                    save_collection="temp-hw",  # Simpan sementara
                    config_id=config_id,
                    append_column_id=True,
                    client=client
                )
                results.append(result)

                # Simpan metrik evaluasi untuk kolom ini
                if result.get("error_metrics"):
                    error_metrics_list.append({
                        "collectionName": collection,
                        "columnName": column,
                        "metrics": {
                            "mae": result["error_metrics"].get("mae"),
                            "rmse": result["error_metrics"].get("rmse"),
                            "mape": result["error_metrics"].get("mape"),
                            "mse": result["error_metrics"].get("mse")
                        }
                    })

               
                
            except Exception as e:
                error_msg = f"Holt-Winter failed for {collection}:{column} → {str(e)}"
                db.forecast_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "errorMessage": error_msg}}
                )
                traceback.print_exc()
                return jsonify({"error": error_msg}), 500
        
        print("[INFO] Combining forecast data...")
        
        # Menggunakan aggregation pipeline untuk menggabungkan data lebih efisien
        pipeline = [
            {"$match": {"config_id": config_id}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$forecast_date"
                        }
                    },
                    "forecast_date": {
                        "$first": {
                            "$dateToString": {
                                "format": "%Y-%m-%d", 
                                "date": "$forecast_date"
                            }
                        }
                    },
                    "parameters": {"$mergeObjects": "$parameters"},
                    "config_id": {"$first": "$config_id"}
                }
            },
            {
                "$addFields": {
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "forecast_date": 1,
                    "timestamp": 1,
                    "config_id": 1,
                    "parameters": 1
                }
            }
        ]
        
        # Eksekusi aggregation
        combined_results = list(db["temp-hw"].aggregate(pipeline))
        
        if combined_results:
            # Hapus data lama untuk config ini
            db["holt-winter"].delete_many({"config_id": config_id})
            
            # Insert data gabungan dengan batch insert
            try:
                db["holt-winter"].insert_many(combined_results, ordered=False)
                print(f"✓ Inserted {len(combined_results)} combined forecast documents using aggregation")
            except Exception as e:
                print(f"Batch insert failed, trying individual inserts: {e}")
                # Fallback: insert satu per satu jika batch gagal
                successful_inserts = 0
                for doc in combined_results:
                    try:
                        db["holt-winter"].insert_one(doc)
                        successful_inserts += 1
                    except Exception as insert_error:
                        print(f"Failed to insert document: {insert_error}")
                print(f"✓ Successfully inserted {successful_inserts}/{len(combined_results)} documents")
        
        # Bersihkan collection temporary sekali di akhir
        db["temp-hw"].delete_many({})

        
        # Update status config dan simpan error metrics
        update_data = {
            "status": "done",
            "error_metrics": error_metrics_list
        }

        # Update status config
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": update_data}
        )

        return jsonify({
        "message": f"Forecasting completed for config: {name}",
        "forecastResultCollection": forecast_coll,
        "results": convert_objectid(results),
        "total_forecast_dates": len(forecast_data),
        "error_metrics": error_metrics_list
         }), 200
        
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {"status": "failed", "errorMessage": error_msg}}
        )
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500