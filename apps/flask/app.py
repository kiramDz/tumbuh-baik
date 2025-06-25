# link tutorial : https://youtu.be/OwxxCibSFKk?si=ogX9zwnWNzlFzfxp
# repo 1 : https://github.com/TheRobBrennan/explore-docker-python-flask-nextjs-typescript/tree/main
# repo 2 : https://github.com/martindavid/flask-nextjs-user-management-example

from flask import Flask, jsonify
from holt_winter.hw_deepseek import run_hw_analysis
from holt_winter.hw_deepseek_opt import run_optimized_hw_analysis
from helpers.objectid_converter import convert_objectid
from holt_winter.summary.bmkg_tanam_summary import generate_tanam_summary

from pymongo import MongoClient

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})

# Helper untuk konversi ObjectId ke string
# def convert_objectid(data):
#     if isinstance(data, list):
#         return [convert_objectid(doc) for doc in data]
#     if isinstance(data, dict):
#         return {
#             key: convert_objectid(value) for key, value in data.items()
#         }
#     if isinstance(data, ObjectId):
#         return str(data)
#     return data

@app.route("/run_optimized_hw_analysis", methods=["GET"])
def run_analysis():
    try:
        result = run_optimized_hw_analysis()
        
        return jsonify({
            "message": "Holt-Winter analysis completed and saved to database.",
            "forecast": convert_objectid(result)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check-mongodb")
def check_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        # client = MongoClient("mongodb://host.docker.internal:27017/")
        db = client["tugas_akhir"]
        collections = db.list_collection_names()
        return jsonify({
            "status": "connected",
            "collections": collections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/run-analysis", methods=["GET"])
# def run_analysis():
    try:
        result = run_hw_analysis()
        return jsonify({
            "message": "Holt-Winter analysis completed and saved to database.",
            "forecast": convert_objectid(result)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-tanam-summary", methods=["GET"])
def generate_summary():
    try:
        result = generate_tanam_summary()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5001)
