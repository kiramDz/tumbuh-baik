from flask import Flask, jsonify
from flask_cors import CORS
from helpers.objectid_converter import convert_objectid
from jobs.run_forecast_from_config import run_forecast_from_config
from pymongo import MongoClient

app = Flask(__name__)
CORS(app, origins="http://localhost:3000") 


@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})

@app.route("/run-forecast", methods=["POST"])
def run_forecast():
    return run_forecast_from_config()


@app.route("/check-mongodb")
def check_mongodb():
    try:
        client = MongoClient("mongodb+srv://hilmi0:8ZqtGJVyMiF8x7YN@cluster0.uuonyyb.mongodb.net/tugas_akhir?retryWrites=true&w=majority&appName=Cluster0")
        # client = MongoClient("mongodb://host.docker.internal:27017/")
        db = client["tugas_akhir"]
        collections = db.list_collection_names()
        return jsonify({
            "status": "connected",
            "collections": collections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5001)
