from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os

from helpers.objectid_converter import convert_objectid
from jobs.run_forecast_from_config import run_forecast_from_config
from jobs.run_lstm import run_lstm_from_config
from pymongo import MongoClient

# === Init Flask ===
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://3.107.238.87"]}})

# === MongoDB Connection ===
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_database("tugas_akhir")


# === Routes ===
@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})


@app.route("/run-forecast", methods=["POST"])
def run_forecast():
    return run_forecast_from_config()

@app.route("/run-lstm", methods=["POST"])
def run_lstm():
    return run_lstm_from_config()


@app.route("/check-mongodb")
def check_mongodb():
    try:
        collections = db.list_collection_names()
        return jsonify({
            "status": "connected",
            "collections": collections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Entry Point ===
if __name__ == "__main__":
    # Jangan pakai debug di production
    app.run(host="0.0.0.0", port=5001)
