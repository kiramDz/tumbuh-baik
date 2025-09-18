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
        db["temp-hw"].delete_many({})
        db["temp-decompose"].delete_many({})  # Tambah: kosongkan temp-decompose
        db["decompose"].delete_many({})

        config = db.forecast_configs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running"}},
            return_document=True
        )
        
        if not config:
            return jsonify({"message": "No pending forecast config found."}), 404
        
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

                # Ambil hasil forecast untuk digabung
                temp_forecasts = list(db["temp-hw"].find({"config_id": config_id}))
                
                for forecast_doc in temp_forecasts:
                    forecast_date = pd.to_datetime(forecast_doc["forecast_date"]).strftime("%Y-%m-%d")

                    
                    if forecast_date not in forecast_data:
                        forecast_data[forecast_date] = {
                            "forecast_date": forecast_date,
                            "timestamp": datetime.now().isoformat(),
                            "config_id": config_id,
                            "parameters": {}
                        }
                    
                    # Tambahkan parameter ke struktur gabungan
                    if "parameters" in forecast_doc:
                        forecast_data[forecast_date]["parameters"].update(
                            forecast_doc["parameters"]
                        )
                
            except Exception as e:
                error_msg = f"Holt-Winter failed for {collection}:{column} → {str(e)}"
                db.forecast_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "errorMessage": error_msg}}
                )
                traceback.print_exc()
                return jsonify({"error": error_msg}), 500
        
        # Simpan hasil gabungan ke collection final
        if forecast_data:
            # Hapus data lama untuk config ini
            db["holt-winter"].delete_many({"config_id": config_id})
            
            # Insert data gabungan
            combined_docs = list(forecast_data.values())
            db["holt-winter"].insert_many(combined_docs)
            
            print(f"✓ Inserted {len(combined_docs)} combined forecast documents")
        
        # Bersihkan collection temporary
        db["temp-hw"].delete_many({})

        temp_decomposes = list(db["temp-decompose"].find({"config_id": config_id}))

        for decompose_doc in temp_decomposes:
            decompose_date = pd.to_datetime(decompose_doc["date"]).strftime("%Y-%m-%d")

            if decompose_date not in decompose_data:
                decompose_data[decompose_date] = {
                    "date": decompose_date,
                    "timestamp": datetime.now().isoformat(),
                    "config_id": config_id,
                    "parameters": {}
                }
            
            if "parameters" in decompose_doc:
                decompose_data[decompose_date]["parameters"].update(
                    decompose_doc["parameters"]
                )
        
        if decompose_data:
            combined_decompose_docs = list(decompose_data.values())
            db["decompose"].insert_many(combined_decompose_docs)
        
        db["temp-decompose"].delete_many({})
        
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
