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

        start_date_str = config.get("startDate")  # ISO format dari MongoDB
        end_date_str = config.get("endDate")

        # Validasi tanggal
        if not start_date_str or not end_date_str:
            error_msg = "startDate and endDate are required in config"
            db.forecast_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "errorMessage": error_msg}}
            )
            return jsonify({"error": error_msg}), 400
        
        # Parse tanggal ke format yang bisa digunakan
        try:
            start_date = pd.to_datetime(start_date_str).date()
            end_date = pd.to_datetime(end_date_str).date()
            print(f"[INFO] Forecast range: {start_date} to {end_date}")
        except Exception as e:
            error_msg = f"Invalid date format: {str(e)}"
            db.forecast_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "errorMessage": error_msg}}
            )
            return jsonify({"error": error_msg}), 400
        

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
                # Jalankan analisis Holt-Winter (sekarang mengembalikan list dari 3 split ratio)
                result_list = run_optimized_hw_analysis(
                    collection_name=collection,
                    target_column=column,
                    save_collection="holt-winter", 
                    config_id=config_id,
                    append_column_id=True,
                    client=client,
                    start_date=start_date,  
                    end_date=end_date
                )
                
                # Tambahkan semua hasil dari 3 split ratio
                results.extend(result_list)

                # Ambil hasil forecast untuk semua split ratio
                temp_forecasts = list(db["holt-winter"].find({"config_id": config_id}))
                
                for forecast_doc in temp_forecasts:
                    forecast_date = pd.to_datetime(forecast_doc["forecast_date"]).strftime("%Y-%m-%d")
                    split_ratio = forecast_doc.get("split_ratio", "unknown")
                    forecast_key = f"{forecast_date}_{split_ratio}"

                    if forecast_key not in forecast_data:
                        forecast_data[forecast_key] = {
                            "forecast_date": forecast_date,
                            "split_ratio": split_ratio,
                            "timestamp": datetime.now().isoformat(),
                            "config_id": config_id,
                            "parameters": {}
                        }
                    
                    if "parameters" in forecast_doc:
                        forecast_data[forecast_key]["parameters"].update(
                            forecast_doc["parameters"]
                        )
                
                # Simpan metrik evaluasi untuk setiap split ratio
                for result in result_list:
                    if result.get("error_metrics"):
                        error_metrics_list.append({
                            "collectionName": collection,
                            "columnName": column,
                            "split_ratio": result.get("split_ratio"),
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