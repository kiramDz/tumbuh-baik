from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask import current_app
from helpers.objectid_converter import convert_objectid
from jobs.run_forecast_from_config import run_forecast_from_config
from pymongo import MongoClient
from bson import ObjectId
import json
from bson.json_util import dumps
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging
import traceback
import tempfile
import shutil
from typing import Dict, Any
from preprocessing.buoys.preprocessing_buoys import (
    DataValidator,
    MongoDataLoader,
    QualityFilter,
    BuoyPreprocessor,
    MongoDataSaver,
    PreprocessingError
)
from preprocessing.nasa.preprocessing_nasa import (
    NasaPreprocessor,
    NasaDataValidator,
    NasaDataSaver,
    NasaPreprocessingError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ORIGINS", "http://localhost:3000"))

# MongoDB connection using environment variables
mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]

# Configure Flask app
app.config['MONGO_DB'] = db
app.config['MONGO_CLIENT'] = mongo_client

# Initialize blueprint - MOVED HERE
preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix="/api/v1")


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
    
@preprocessing_bp.route("/preprocess/<collection_name>", methods=["POST"])
def preprocess_dataset(collection_name):
    """Preprocess a dataset using the new class-based system"""
    db = request.app.config['MONGO_DB']
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
        
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
        
        # Create a status document to track progress
        preprocessing_status = {
            "job_id": job_id,
            "collection": collection_name,
            "status": "processing",
            "startTime": datetime.now(),
            "steps": [
                {"name": "started", "completed": True, "timestamp": datetime.now()}
            ]
        }
        
        # Save status to MongoDB
        db["preprocessing_jobs"].insert_one(preprocessing_status)
        
        # Get preprocessing options from request body
        options = request.json or {}
        
        # Extract location code from collection name if available
        location_code = None
        if collection_name.startswith("buoys_"):
            location_parts = collection_name.split("_")
            if len(location_parts) >= 3:
                location_code = location_parts[-1]
        
        try:
            # Update status to indicate validation started
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "validating_data", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Validate the dataset
            validator = DataValidator()
            validation_result = validator.validate_dataset(db, collection_name)
            
            if validation_result and not validation_result.get('valid', True):
                return jsonify({
                    "message": "Dataset validation failed",
                    "job_id": job_id,
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }), 400
            
            # Update status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "preprocessing_started", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Initialize preprocessor
            preprocessor = BuoyPreprocessor(db, collection_name, location_code)
            
            # Run preprocessing
            preprocessing_results = preprocessor.preprocess(options)
            
            # Update status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "preprocessing_completed", "completed": True, "timestamp": datetime.now()}}}
            )
            
            # Save processed data back to MongoDB
            saver = MongoDataSaver()
            save_results = saver.save_processed_data(db, preprocessing_results, collection_name)
            
            # Update job status
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed", 
                        "endTime": datetime.now(), 
                        "metadata": save_results
                    },
                    "$push": {"steps": {"name": "results_saved", "completed": True, "timestamp": datetime.now()}}
                }
            )
            
            # Return success response
            return jsonify({
                "message": "Preprocessing completed successfully",
                "job_id": job_id,
                "collection": collection_name,
                "processedCollections": save_results.get("processedCollections", []),
                "recordCounts": save_results.get("recordCounts", {}),
                "status": "preprocessed",
                "warnings": preprocessing_results.get("warnings", [])
            }), 200
            
        except PreprocessingError as pe:
            error_details = str(pe)
            
            # Update status with error
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {"status": "failed", "endTime": datetime.now(), "error": error_details},
                    "$push": {"steps": {"name": "processing_failed", "completed": False, "timestamp": datetime.now(), "error": error_details}}
                }
            )
            
            return jsonify({
                "message": f"Preprocessing error: {error_details}",
                "job_id": job_id,
                "status": "failed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500

@preprocessing_bp.route("/preprocess/status/<job_id>", methods=["GET"])
def get_preprocessing_status(job_id):
    """Get status of a preprocessing job"""
    db = request.app.config['MONGO_DB']
    
    try:
        # Find job in database
        job = db["preprocessing_jobs"].find_one({"job_id": job_id})
        
        if not job:
            return jsonify({"message": f"Job with ID '{job_id}' not found"}), 404
            
        # Format response
        response = {
            "job_id": job["job_id"],
            "collection": job["collection"],
            "status": job["status"],
            "started": job.get("startTime"),
            "completed": job.get("endTime"),
            "steps": job.get("steps", []),
        }
        
        # Add error details if job failed
        if job["status"] == "failed":
            response["error"] = job.get("error")
            
        # Add metadata if job completed
        if job["status"] == "completed":
            response["metadata"] = job.get("metadata")
            
        return jsonify({"message": "Success", "data": response}), 200
        
    except Exception as e:
        logger.error(f"Error fetching job status: {str(e)}")
        return jsonify({"message": f"Server error: {str(e)}"}), 500


@preprocessing_bp.route("/preprocess/nasa/<collection_name>", methods=["POST"])
def preprocess_nasa_dataset(collection_name):
    """Preprocessing a NASA POWER dataset"""
    db = current_app.config['MONGO_DB']
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({"message": f"Collection '{collection_name}' not found"}), 404
        
        # Get total records
        total_records = db[collection_name].count_documents({})
        logger.info(f"Starting preprocessing for NASA dataset '{collection_name}' with {total_records} records")
        
        # Generate a job ID to track this preprocessing task
        job_id = str(ObjectId())
        
        # Create a status document to track progress
        preprocessing_status = {
            "job_id": job_id,
            "collection": collection_name,
            "type": "nasa-power",
            "status": "processing",
            "startTime": datetime.now(),
            "totalRecords": total_records,
            "processedRecords": 0,
            "steps": [
                {"name": "started", "completed": True, "timestamp": datetime.now()}
            ]
        }
        
        # Save status to MongoDB
        db["preprocessing_jobs"].insert_one(preprocessing_status)
        
        # GET preprocessing options from request body
        options = request.json or {}
        
        try:
            # Update status to indicate validation started
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {"$push": {"steps": {"name": "validating_data", "completed": True, "timestamp": datetime.now()}}}
            )
            # Validate the dataset
            validator = NasaDataValidator()
            validation_result = validator.validate_dataset(db, collection_name)
            
            if validation_result and not validation_result.get('valid', True):
                db['preprocessing_jobs'].update_one(
                    {"job_id": job_id},
                    {"$set": {"status": "failed", "error": "Validation failed"}}
                )
                return jsonify({
                    "message": "Dataset validation failed",
                    "job_id": job_id,
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }), 400
                
            # Update status
            db['preprocessing_jobs'].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "loading_data", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "loading_data"}
                }
            )
            
            # Initialize preprocessor
            preprocessor = NasaPreprocessor(db, collection_name)
            
            # Update status for preprocessing started
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_started", "completed": True, "timestamp": datetime.now()}},
                    "$set": {"status": "processing", "currentStep": "preprocessing"}
                }
            )
            # Run preprocessing
            preprocessing_results = preprocessor.preprocess(options)
            
            # Update status for preprocessing completion
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$push": {"steps": {"name": "preprocessing_completed", "completed": True, "timestamp": datetime.now()}},
                    "$set": {
                        "processedRecords": preprocessing_results.get("recordCount", total_records),
                        "status": "saving_results",
                        "currentStep": "saving_results"
                    }
                }
            )
            # Final status update
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed", 
                        "endTime": datetime.now(), 
                        "metadata": preprocessing_results
                    },
                    "$push": {"steps": {"name": "results_saved", "completed": True, "timestamp": datetime.now()}}
                }
            )
            # Return success response
            return jsonify({
                "message": "NASA POWER preprocessing completed successfully",
                "job_id": job_id,
                "collection": collection_name,
                "cleanedCollection": preprocessing_results.get("cleanedCollection"),
                "preprocessedCollection": f"{collection_name}_cleaned",
                "status": "preprocessed",
                "recordCount": preprocessing_results.get("recordCount", 0),
                "warnings": preprocessing_results.get("warnings", [])
            }), 200
        except NasaPreprocessingError as pe:
            error_details = str(pe)
            logger.error(f"NasaPreprocessingError: {error_details}")
            
            # Update status with error
            db["preprocessing_jobs"].update_one(
                {"job_id": job_id},
                {
                    "$set": {"status": "failed", "endTime": datetime.now(), "error": error_details},
                    "$push": {"steps": {"name": "processing_failed", "completed": False, "timestamp": datetime.now(), "error": error_details}}
                }
            )
            
            return jsonify({
                "message": f"Preprocessing error: {error_details}",
                "job_id": job_id,
                "status": "failed"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in NASA preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"message": f"Server error: {str(e)}"}), 500


# TAMBAHKAN BARIS INI
app.register_blueprint(preprocessing_bp)


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
    # # Jangan pakai debug di production
    # app.run(host="0.0.0.0", port=5001)