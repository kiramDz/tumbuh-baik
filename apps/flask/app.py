from flask import Flask, jsonify, request
from flask_cors import CORS
from helpers.objectid_converter import convert_objectid
from jobs.run_forecast_from_config import run_forecast_from_config
from pymongo import MongoClient
from bson import ObjectId
import json
from bson.json_util import dumps
import os
from dotenv import load_dotenv
from datetime import datetime  # Add this import

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ORIGINS", "http://localhost:3000"))

# MongoDB connection using environment variables
mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]

@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})

@app.route("/run-forecast", methods=["POST"])
def run_forecast():
    return run_forecast_from_config()

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

# Dataset routes
@app.route("/api/v1/datasets", methods=["GET"])
def get_all_datasets():
    """Get all datasets metadata"""
    try:
        # Check if DatasetMeta collection exists
        if "dataset_meta" in db.list_collection_names():
            datasets = list(db["dataset_meta"].find().sort("uploadDate", -1))
        elif "DatasetMeta" in db.list_collection_names():
            datasets = list(db["DatasetMeta"].find().sort("uploadDate", -1))
        else:
            # If no metadata collection exists, create a simple one based on available collections
            all_collections = db.list_collection_names()
            # Filter out system collections
            exclude_prefixes = ['system.', 'user', 'session', 'account']
            filtered_collections = [coll for coll in all_collections 
                                   if not any(coll.startswith(prefix) for prefix in exclude_prefixes)]
            
            datasets = []
            for coll in filtered_collections:
                # Create a simple metadata entry for each collection
                collection_type = "buoy" if coll.startswith("buoys_") else "general"
                datasets.append({
                    "collectionName": coll,
                    "title": coll.replace("_", " ").title(),
                    "description": f"Data collection for {coll}",
                    "dataType": collection_type,
                    "uploadDate": datetime.now(),
                    "generated": False
                })
        
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps({"data": datasets})), 200
    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@app.route("/api/v1/dataset/<collection_name>", methods=["GET"])
def get_dataset_by_collection(collection_name):
    """Get dataset by collection name with pagination"""
    try:
        # Get query parameters
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("pageSize", 10))
        sort_by = request.args.get("sortBy", "Date") 
        sort_order = request.args.get("sortOrder", "desc")
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
            
        # Get metadata for the collection (if it exists)
        meta = None
        if "dataset_meta" in db.list_collection_names():
            meta = db["dataset_meta"].find_one({"collectionName": collection_name})
        elif "DatasetMeta" in db.list_collection_names():
            meta = db["DatasetMeta"].find_one({"collectionName": collection_name})
            
        # If no metadata, create a simple one
        if not meta:
            # Determine data type based on collection name
            data_type = "buoy" if collection_name.startswith("buoys_") else "general"
            
            # Extract more info for buoy collections
            location = None
            if data_type == "buoy":
                location_parts = collection_name.split("_")
                if len(location_parts) >= 3:
                    location = location_parts[-1]
            
            meta = {
                "collectionName": collection_name,
                "title": collection_name.replace("_", " ").title(),
                "description": f"Data collection for {collection_name}",
                "dataType": data_type,
                "location": location
            }
            
        # Create sort query
        sort_direction = -1 if sort_order == "desc" else 1
        # Ensure the sort field exists in the collection
        sample_doc = db[collection_name].find_one()
        if sample_doc and sort_by not in sample_doc and sort_by != "_id":
            # If the specified sort_by doesn't exist, default to _id
            sort_by = "_id"
            
        sort_query = [(sort_by, sort_direction)]
        
        # Count total documents
        total_data = db[collection_name].count_documents({})
        
        # Get paginated data
        data = list(db[collection_name].find()
                   .sort(sort_query)
                   .skip((page - 1) * page_size)
                   .limit(page_size))
        
        # Get collection stats
        stats = {
            "name": collection_name,
            "total_records": total_data,
            "fields": list(sample_doc.keys()) if sample_doc else []
        }
        
        # Prepare response
        response = {
            "message": "Success",
            "data": {
                "meta": meta,
                "stats": stats,
                "items": data,
                "total": total_data,
                "currentPage": page,
                "totalPages": (total_data + page_size - 1) // page_size,
                "pageSize": page_size,
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
        }
        
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps(response)), 200
        
    except Exception as e:
        print(f"Error fetching dataset: {str(e)}")
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@app.route("/api/v1/datasets/<collection_name>/<object_id>", methods=["GET"])
def get_dataset_document_by_id(collection_name, object_id):
    """Get a specific document from a dataset collection by its ObjectID"""
    try:
        # Validate ObjectId
        if not ObjectId.is_valid(object_id):
            return jsonify({"message": "Invalid ObjectID format"}), 400
            
        # Get the document
        document = db[collection_name].find_one({"_id": ObjectId(object_id)})
        
        if not document:
            return jsonify({"message": "Document not found"}), 404
            
        # Convert ObjectId to string for JSON serialization
        return json.loads(dumps({"data": document})), 200
        
    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500

# Preprocessing endpoint
@app.route("/api/v1/preprocess/<collection_name>", methods=["POST"])
def preprocess_dataset(collection_name):
    """Preprocess a dataset"""
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
            
        # Handle different dataset types
        if collection_name.startswith("buoys_"):
            # Get location code from collection name for buoy datasets
            location_parts = collection_name.split("_")
            location_code = location_parts[-1].upper() if len(location_parts) >= 3 else None
            
            data_type = "buoy"
            # Here you would call your buoys preprocessing function
            # from preprocessing_buoys import preprocess_buoy_data
            # result = preprocess_buoy_data(...)
        else:
            # Handle other dataset types
            data_type = "general"
            location_code = None
            
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
            
        # For now, just return a message
        return jsonify({
            "message": f"Preprocessing request received for {collection_name}",
            "collection": collection_name,
            "dataType": data_type,
            "location": location_code,
            "status": "pending",
            "job_id": job_id
        }), 202
        
    except Exception as e:
        return jsonify({"message": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)