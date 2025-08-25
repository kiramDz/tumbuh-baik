from pymongo import MongoClient
from datetime import datetime
import time
import traceback
import os
from flask import jsonify
from dotenv import load_dotenv
from lstm.lstm_dynamic_2 import run_optimized_lstm_analysis


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

def update_lstm_status(config_id, status, error_message=None, error_metrics=None):
    """Update LSTM execution status in lstm_configs collection only"""
    try:
        from bson import ObjectId
        
        update_doc = {
            "status": status,
            "updated_at": datetime.now(),
            "last_status_update": datetime.now().isoformat()
        }
        
        if error_message:
            update_doc["error_message"] = error_message
        else:
            # Clear error message if status is successful
            if status in ["running", "done", "completed"]:
                update_doc["error_message"] = None
            
        if error_metrics:
            update_doc["error_metrics"] = error_metrics
            
        # Add completion timestamp for successful runs
        if status in ["done", "completed"]:
            update_doc["completed_at"] = datetime.now()
            
        # Update only lstm_configs collection
        result = db["lstm_configs"].update_one(
            {"_id": ObjectId(config_id)},
            {"$set": update_doc}
        )
        
        if result.modified_count > 0:
            print(f"✓ LSTM status updated: {config_id} → {status}")
        else:
            print(f"⚠️ No document updated for config_id: {config_id}")
        
    except Exception as e:
        print(f"❌ Error updating LSTM status: {str(e)}")

def get_pending_lstm_config():
    """Get pending config and update status to running"""
    try:
        # Cari config yang pending di lstm_configs
        config = db.lstm_configs.find_one({"status": "pending"})
        
        if not config:
            print("📋 No pending LSTM config found")
            return None
            
        config_id = str(config["_id"])
        print(f"📋 Found pending LSTM config: {config_id}")
        
        # Update status ke running di lstm_configs collection
        update_lstm_status(config_id, "running")
        
        # Update last_execution timestamp
        db.lstm_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {"last_execution": datetime.now()}}
        )
        
        return config
        
    except Exception as e:
        print(f"❌ Error getting pending config: {str(e)}")
        return None

def run_lstm_from_config():
    try:
        # Kosongkan collection temp-lstm di awal
        db["temp-lstm"].delete_many({})

        # Get pending config tanpa mengubah status di lstm_configs
        config = get_pending_lstm_config()
        
        if not config:
            return jsonify({"message": "No pending LSTM config found."}), 404

        config_id = str(config["_id"])
        
        # Kosongkan collection lstm-forecast di awal (tanpa perlu pengecekan)
        db["lstm-forecast"].delete_many({})
        
        # Ambil info kolom yang akan dianalisis
        name = config.get("name", f"lstm_forecast_{int(time.time())}")
        columns = config.get("columns", [])
        forecast_coll = config.get("forecastResultCollection", "lstm-forecast")

        # Validasi kolom
        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            is_valid, error_msg = is_valid_column(collection, column, client)
            if not is_valid:
                # Update status ke failed di lstm_configs collection
                update_lstm_status(config_id, "failed", error_msg)
                return jsonify({"error": error_msg}), 400
        
        results = []
        forecast_data = {}  # Untuk menyimpan semua forecast berdasarkan tanggal
        error_metrics_list = []

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]
            
            print(f"[INFO] Processing LSTM for {collection} - {column}")
            
            try:
                # Jalankan analisis LSTM
                result = run_optimized_lstm_analysis(
                    collection_name=collection,
                    target_column=column,
                    save_collection="temp-lstm",  # Simpan sementara
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
                        "model_type": "LSTM",
                        "metrics": {
                            "mae": result["error_metrics"].get("mae"),
                            "rmse": result["error_metrics"].get("rmse"),
                            "mape": result["error_metrics"].get("mape")
                        },
                        "model_params": result.get("model_params", {})
                    })
                
                # Ambil hasil forecast untuk digabung
                temp_forecasts = list(db["temp-lstm"].find({"config_id": config_id}))
                
                for forecast_doc in temp_forecasts:
                    forecast_date = forecast_doc["forecast_date"]
                    
                    if forecast_date not in forecast_data:
                        forecast_data[forecast_date] = {
                            "forecast_date": forecast_date,
                            "timestamp": datetime.now().isoformat(),
                            "config_id": config_id,
                            "model_type": "LSTM",
                            "parameters": {}
                        }
                    
                    # Tambahkan parameter ke struktur gabungan
                    if "parameters" in forecast_doc:
                        forecast_data[forecast_date]["parameters"].update(
                            forecast_doc["parameters"]
                        )
                
            except Exception as e:
                error_msg = f"LSTM failed for {collection}:{column} → {str(e)}"
                
                # Update status ke failed di lstm_configs collection
                update_lstm_status(config_id, "failed", error_msg)
                
                traceback.print_exc()
                return jsonify({"error": error_msg}), 500
        
        # Simpan hasil gabungan ke collection final
        if forecast_data:
            # Hapus data lama untuk config ini
            db["lstm-forecast"].delete_many({"config_id": config_id})
            
            # Insert data gabungan
            combined_docs = list(forecast_data.values())
            db["lstm-forecast"].insert_many(combined_docs)
            
            print(f"✓ Inserted {len(combined_docs)} combined LSTM forecast documents")
        
        # Bersihkan collection temporary
        db["temp-lstm"].delete_many({})
        
        # Update status ke done di lstm_configs collection dengan error metrics
        update_lstm_status(config_id, "done", None, error_metrics_list)

        return jsonify({
            "message": f"LSTM Forecasting completed for config: {name}",
            "config_id": config_id,
            "forecastResultCollection": forecast_coll,
            "model_type": "LSTM",
            "results": convert_objectid(results),
            "total_forecast_dates": len(forecast_data),
            "error_metrics": error_metrics_list,
            "status_collection": "lstm_configs"
        }), 200
        
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        if 'config_id' in locals():
            # Update status ke failed di lstm_configs collection
            update_lstm_status(config_id, "failed", error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

def get_lstm_status(config_id):
    """Get LSTM execution status from lstm_configs collection"""
    try:
        from bson import ObjectId
        status_doc = db["lstm_configs"].find_one({"_id": ObjectId(config_id)})
        if status_doc:
            return convert_objectid(status_doc)
        return None
    except Exception as e:
        print(f"❌ Error getting LSTM status: {str(e)}")
        return None

def get_all_lstm_statuses():
    """Get all LSTM execution statuses from lstm_configs collection"""
    try:
        statuses = list(db["lstm_configs"].find().sort("updated_at", -1))
        return convert_objectid(statuses)
    except Exception as e:
        print(f"❌ Error getting all LSTM statuses: {str(e)}")
        return []

# Function untuk testing langsung
def test_lstm_forecast():
    """Function untuk testing LSTM forecast secara langsung"""
    try:
        # Test dengan data dummy config
        test_config = {
            "_id": "test_lstm_config",
            "name": "Test LSTM Forecast",
            "columns": [
                {"collectionName": "bmkg-data", "columnName": "RR"},
                {"collectionName": "bmkg-data", "columnName": "RH_AVG"}
            ],
            "forecastResultCollection": "lstm-forecast"
        }

        # Insert test config ke lstm_configs
        db.lstm_configs.delete_many({"_id": "test_lstm_config"})
        db.lstm_configs.insert_one({**test_config, "status": "pending"})

        # Run LSTM forecast
        result = run_lstm_from_config()
        print("LSTM Test Result:", result)
        
        # Check final status
        final_status = get_lstm_status("test_lstm_config")
        print("Final LSTM Status:", final_status)
        
        return result
        
    except Exception as e:
        print(f"Error in LSTM test: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Untuk testing langsung
    print("🧪 Starting LSTM test...")
    result = test_lstm_forecast()
    if result:
        print("✅ LSTM test completed successfully!")
    else:
        print("❌ LSTM test failed!")
    print("🏁 Test finished.")