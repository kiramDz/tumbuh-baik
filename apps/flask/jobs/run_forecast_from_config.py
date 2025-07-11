# jobs/run_forecast_from_config.py

import os
from flask import jsonify
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import traceback
import time

from holt_winter.hw_dynamic import run_optimized_hw_analysis
from helpers.objectid_converter import convert_objectid

# Ganti sesuai URL MongoDB Atlas kamu
# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise ValueError("No MONGODB_URI set in environment variables!")

client = MongoClient(MONGO_URI)
db = client["tugas_akhir"] 

def run_forecast_from_config():
    try:
        config = db.forecast_configs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running"}},
            return_document=True
        )

        if not config:
            return jsonify({"message": "No pending forecast config found."}), 404

        # Ambil info kolom yang akan dianalisis
        name = config.get("name", f"forecast_{int(time.time())}")
        columns = config.get("columns", [])
        forecast_coll = config.get("forecastResultCollection")

        results = []

        for item in columns:
            collection = item["collectionName"]
            column = item["columnName"]

            print(f"[INFO] Processing {collection} - {column}")

            try:
                result = run_optimized_hw_analysis(
                collection_name=collection,
                target_column=column,
                save_collection="holt-winter",  # atau tetap dari config
                config_id=str(config["_id"]),
                append_column_id=True
                )
                results.append(result)

            except Exception as e:
                error_msg = f"Holt-Winter failed for {collection}:{column} → {str(e)}"
                db.forecast_configs.update_one(
                    {"_id": config["_id"]},
                    {"$set": {"status": "failed", "errorMessage": error_msg}}
                )
                traceback.print_exc()
                return jsonify({"error": error_msg}), 500

        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {"status": "done"}}
        )

        return jsonify({
            "message": f"Forecasting completed for config: {name}",
            "forecastResultCollection": forecast_coll,
            "results": convert_objectid(results)
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
