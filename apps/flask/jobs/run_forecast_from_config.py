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
    
    try:
        sample_doc = client["tugas_akhir"][collection_name].find_one({column_name: {"$exists": True}})
        if sample_doc and not isinstance(sample_doc[column_name], (int, float)):
            return False, f"Column {column_name} contains non-numeric data"
        return True, None
    except Exception as e:
        return False, f"Error checking column {column_name}: {str(e)}"

def run_forecast_from_config():
    try:
        # Kosongkan collections di awal
        db["temp-hw"].delete_many({})
        db["temp-decompose"].delete_many({})
        db["decompose"].delete_many({})
        db["holt-winter"].delete_many({})
        db["hw-historical"].delete_many({})

        config = db.forecast_configs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running"}},
            return_document=True
        )
        
        if not config:
            return jsonify({"message": "No pending forecast config found."}), 404
        
        name = config.get("name", f"forecast_{int(time.time())}")
        columns = config.get("columns", [])
        forecast_coll = config.get("forecastResultCollection")
        config_id = str(config["_id"])

        start_date_str = config.get("startDate")
        end_date_str = config.get("endDate")

        if not start_date_str or not end_date_str:
            error_msg = "startDate and endDate are required in config"
            db.forecast_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "errorMessage": error_msg}}
            )
            return jsonify({"error": error_msg}), 400
        
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
        error_metrics_list = []

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            
            print(f"[INFO] Processing {collection} - {column}")
            
            try:
                result_list = run_optimized_hw_analysis(
                    collection_name=collection,
                    target_column=column,
                    config_id=config_id,
                    client=client,
                    start_date=start_date,  
                    end_date=end_date
                )
                
                results.extend(result_list)
                
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
        
        # Combine multi-parameter per tanggal untuk forecast
        forecast_combined = {}
        all_forecasts = list(db["holt-winter"].find({"config_id": config_id}))
        
        for doc in all_forecasts:
            date_str = pd.to_datetime(doc["forecast_date"]).strftime("%Y-%m-%d")
            split_ratio = doc.get("split_ratio", "unknown")
            key = f"{date_str}_{split_ratio}"
            
            if key not in forecast_combined:
                forecast_combined[key] = {
                    "forecast_date": date_str,
                    "split_ratio": split_ratio,
                    "config_id": config_id,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {}
                }
            
            if "parameters" in doc:
                forecast_combined[key]["parameters"].update(doc["parameters"])
        
        # Combine multi-parameter per tanggal untuk historical (train + validation)
        historical_combined = {}
        all_historical = list(db["hw-historical"].find({"config_id": config_id}))
        
        for doc in all_historical:
            date_str = pd.to_datetime(doc["date"]).strftime("%Y-%m-%d")
            split_ratio = doc.get("split_ratio", "unknown")
            data_type = doc.get("data_type", "unknown")
            key = f"{date_str}_{split_ratio}_{data_type}"
            
            if key not in historical_combined:
                historical_combined[key] = {
                    "date": date_str,
                    "split_ratio": split_ratio,
                    "data_type": data_type,
                    "config_id": config_id,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {}
                }
            
            if "parameters" in doc:
                historical_combined[key]["parameters"].update(doc["parameters"])
        
        # Replace dengan data combined
        if forecast_combined:
            db["holt-winter"].delete_many({"config_id": config_id})
            db["holt-winter"].insert_many(list(forecast_combined.values()))
            print(f"✓ Inserted {len(forecast_combined)} combined forecast documents")
        
        if historical_combined:
            db["hw-historical"].delete_many({"config_id": config_id})
            db["hw-historical"].insert_many(list(historical_combined.values()))
            print(f"✓ Inserted {len(historical_combined)} combined historical documents")
        
        # Bersihkan collection temporary
        db["temp-hw"].delete_many({})
        db["temp-decompose"].delete_many({})
        
        # Update status config
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {
                "status": "done",
                "error_metrics": error_metrics_list
            }}
        )

        return jsonify({
            "message": f"Forecasting completed for config: {name}",
            "forecastResultCollection": forecast_coll,
            "results": convert_objectid(results),
            "total_forecast_dates": len(forecast_combined),
            "total_historical_records": len(historical_combined),
            "error_metrics": error_metrics_list
        }), 200
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        traceback.print_exc()
        if config:
            db.forecast_configs.update_one(
                {"_id": config["_id"]},
                {"$set": {"status": "failed", "errorMessage": error_msg}}
            )
        return jsonify({"error": error_msg}), 500