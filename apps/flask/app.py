# link tutorial : https://youtu.be/OwxxCibSFKk?si=ogX9zwnWNzlFzfxp
# repo 1 : https://github.com/TheRobBrennan/explore-docker-python-flask-nextjs-typescript/tree/main
# repo 2 : https://github.com/martindavid/flask-nextjs-user-management-example

from flask import Flask, jsonify
from holt_winter.hw_analysis import run_hw_analysis
from pymongo import MongoClient

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})

@app.route("/test-db", methods=["GET"])
def test_db():
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://mongodb:27017/")
        db = client["tugas_akhir"]
        collection = db["bmkg-api"]
        
        count = collection.count_documents({})
        sample = list(collection.find({}).limit(1))
        
        return jsonify({
            "status": "connected", 
            "document_count": count,
            "sample_document": sample
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# by deepseek
@app.route("/check-mongodb")
def check_mongodb():
    try:
        client = MongoClient("mongodb://host.docker.internal:27017/")
        db = client["tugas_akhir"]
        collections = db.list_collection_names()
        return jsonify({
            "status": "connected",
            "collections": collections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run-analysis", methods=["GET"])
def run_analysis():
    try:
        run_hw_analysis()
        return jsonify({"message": "Holt-Winter analysis completed and saved to database."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5000)