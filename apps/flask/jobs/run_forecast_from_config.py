from pymongo import MongoClient
from datetime import datetime
import time
import traceback
import os
from flask import jsonify
from dotenv import load_dotenv
from holt_winter.hw_dynamic import run_optimized_hw_analysis
from holt_winter.summary.monthly_summary_rev4 import generate_monthly_summary

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


def run_forecast_from_config():
    try:

        # Kosongkan collection temp-hw di awal
        db["temp-hw"].delete_many({})

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
        
        results = []
        forecast_data = {}  # Untuk menyimpan semua forecast berdasarkan tanggal
        
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
                
                # Ambil hasil forecast untuk digabung
                temp_forecasts = list(db["temp-hw"].find({"config_id": config_id}))
                
                for forecast_doc in temp_forecasts:
                    forecast_date = forecast_doc["forecast_date"]
                    
                    # Inisialisasi struktur jika belum ada
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
        
        # Panggil function generate_monthly_summary langsung
        summary_result = generate_monthly_summary(config_id, client)

        # Periksa apakah summary berhasil dibuat
        if not isinstance(summary_result, dict):
            raise ValueError("Invalid summary result format")
    
        # Update status config
        update_data = {"status": "done"}
        if summary_result.get("success"):
            update_data.update({
                "summary_generated": True,
                "summary_months": summary_result.get("summaries_generated", 0)
            })
        else:
            update_data["summary_error"] = summary_result.get("error", "Unknown error")

        # Update status config
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {"status": "done"}}
        )

        return jsonify({
        "message": f"Forecasting completed for config: {name}",
        "forecastResultCollection": forecast_coll,
        "results": convert_objectid(results),
        "total_forecast_dates": len(forecast_data),
        "summary_result": summary_result  # Langsung passing dictionary
         }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500