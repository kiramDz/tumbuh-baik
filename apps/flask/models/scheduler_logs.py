import os
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger(__name__)

VALID_STATUSES = {"running", "success", "failed", "partial"}
VALID_TRIGGER_TYPES = {"cron", "manual", "api"}
VALID_TASK_NAMES = {"nasa_refresh", "bmkg_preprocess"}
VALID_TASK_STATUSES = {"success", "failed", "skipped", "running"}

class SchedulerLog:
    """
    MongoDB Model logic for scheduler_logs collection using PyMongo.
    """
    
    @staticmethod
    def _get_db() -> Database:
        """Helper to get MongoDB database instance."""
        # Tries to get URI from env, otherwise fallback to the provided one. 
        # (It is recommended to rely strictly on environment variables in production)
        mongo_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "tugas_akhir")
        client = MongoClient(mongo_uri)
        return client[db_name]

    @staticmethod
    def _get_collection(db: Database) -> Collection:
        """Helper to get collection and ensure indexes are created."""
        collection = db["scheduler_logs"]
        
        # Ensure Indexes
        collection.create_index([("executedAt", DESCENDING)])
        collection.create_index([("status", ASCENDING)])
        collection.create_index([("triggeredBy", ASCENDING)])
        collection.create_index([("executedAt", DESCENDING), ("status", ASCENDING)])
        
        return collection
    
    @staticmethod
    def create_log(triggered_by: str = "cron") -> str:
        """
        Create a new log entry with status 'running'.
        
        Args:
            triggered_by: Source of the trigger ("cron", "manual", "api").
            
        Returns:
            str: The stringified ObjectId of the newly created log.
            
        Raises:
            ValueError: If triggered_by is invalid.
        """
        if triggered_by not in VALID_TRIGGER_TYPES:
            raise ValueError(f"Invalid triggered_by. Must be one of: {VALID_TRIGGER_TYPES}")

        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        now = datetime.now(timezone.utc)
        
        log_document = {
            "executedAt": now,
            "completedAt": None,
            "status": "running",
            "duration": None,
            "triggeredBy": triggered_by,
            "tasks": [],
            "totalDatasets": 0,
            "datasetsUpdated": 0,
            "errors": [],
            "metadata": {
                "nasaApiVersion": os.getenv("NASA_API_VERSION", "unknown"),
                "pythonVersion": "3.10+",
                "system": "ubuntu-server"
            },
            "createdAt": now,
            "updatedAt": now
        }
        
        try:
            result = collection.insert_one(log_document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to create scheduler log: {e}")
            raise
    
    @staticmethod
    def update_log(log_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing log (status, tasks, errors, etc.).
        
        Args:
            log_id: The string ID of the log to update.
            updates: Dictionary of fields to update using PyMongo update operators ($set, $push, etc.)
                     If regular dict is passed, it will be wrapped in $set.
                     
        Returns:
            bool: True if updated successfully.
        """
        if not ObjectId.is_valid(log_id):
            raise ValueError("Invalid log_id format")

        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        # Validate status if it's being updated
        if "status" in updates and updates["status"] not in VALID_STATUSES:
             raise ValueError(f"Invalid status. Must be one of: {VALID_STATUSES}")
             
        # Automatically update updatedAt timestamp
        now = datetime.now(timezone.utc)
        
        update_query = {}
        if any(k.startswith('$') for k in updates.keys()):
            # User passed MongoDB operators ($set, $push, etc)
            update_query = updates
            if "$set" not in update_query:
                update_query["$set"] = {}
            update_query["$set"]["updatedAt"] = now
        else:
            # Wrap in $set
            update_query = {"$set": {**updates, "updatedAt": now}}

        try:
            result = collection.update_one(
                {"_id": ObjectId(log_id)},
                update_query
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update scheduler log {log_id}: {e}")
            raise
    
    @staticmethod
    def complete_log(log_id: str, status: str = "success", datasets_updated: int = 0, total_datasets: int = 0) -> bool:
        """
        Mark log as completed, calculate duration, and set final status.
        
        Args:
            log_id: The string ID of the log.
            status: Final status ("success", "failed", "partial").
            datasets_updated: Actual success count.
            total_datasets: Expected count.
            
        Returns:
            bool: True if successfully marked as completed.
        """
        if status not in VALID_STATUSES or status == "running":
             raise ValueError(f"Invalid final status. Must be one of: success, failed, partial")
             
        if not ObjectId.is_valid(log_id):
            raise ValueError("Invalid log_id format")

        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        # Get the log to calculate duration
        log = collection.find_one({"_id": ObjectId(log_id)})
        if not log:
            raise ValueError(f"Log with ID {log_id} not found")
            
        now = datetime.now(timezone.utc)
        executed_at = log.get("executedAt")
        
        # Ensure executedAt is valid and not in the future relative to now
        if not executed_at:
            executed_at = now
            
        # Ensure dates are timezone aware for subtraction
        if executed_at.tzinfo is None:
            executed_at = executed_at.replace(tzinfo=timezone.utc)
            
        duration = max(0, (now - executed_at).total_seconds())

        # Validate datasets count
        datasets_updated = min(datasets_updated, total_datasets)

        updates = {
            "completedAt": now,
            "status": status,
            "duration": duration,
            "totalDatasets": total_datasets,
            "datasetsUpdated": datasets_updated,
            "updatedAt": now
        }

        try:
            result = collection.update_one(
                {"_id": ObjectId(log_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to complete scheduler log {log_id}: {e}")
            raise
        
    @staticmethod
    def get_latest() -> Optional[Dict[str, Any]]:
        """
        Get the most recent log entry.
        
        Returns:
            Dict representing the document, or None if not found.
        """
        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        try:
            log = collection.find_one({}, sort=[("executedAt", DESCENDING)])
            if log:
                log["_id"] = str(log["_id"])
            return log
        except Exception as e:
            logger.error(f"Failed to get latest log: {e}")
            return None
        
    @staticmethod
    def get_logs(limit: int = 10, offset: int = 0, status_filter: str = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated logs with optional status filter.
        
        Args:
            limit: Number of documents to return.
            offset: Number of documents to skip.
            status_filter: Optional status to filter by.
            
        Returns:
            Tuple of (List of log dictionaries, total count matching filter).
        """
        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        query = {}
        if status_filter:
            if status_filter in VALID_STATUSES:
                query["status"] = status_filter
            else:
                logger.warning(f"Ignored invalid status filter: {status_filter}")

        try:
            cursor = collection.find(query).sort("executedAt", DESCENDING).skip(offset).limit(limit)
            logs = []
            for log in cursor:
                log["_id"] = str(log["_id"])
                logs.append(log)
                
            total = collection.count_documents(query)
            return logs, total
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return [], 0

    @staticmethod
    def get_success_rate(days: int = 30) -> float:
        """
        Calculate success rate percentage for the last N days.
        Ignores 'running' status.
        
        Args:
            days: Time window in days.
            
        Returns:
            float: Percentage of successful or partial runs (0.0 to 100.0).
        """
        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pipeline = [
            {"$match": {
                "executedAt": {"$gte": cutoff_date},
                "status": {"$ne": "running"}
            }},
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1}
            }}
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            
            total = 0
            success_count = 0
            
            for res in results:
                total += res["count"]
                if res["_id"] in ("success", "partial"):
                    success_count += res["count"]
                    
            if total == 0:
                return 0.0
                
            return round((success_count / total) * 100, 2)
        except Exception as e:
            logger.error(f"Failed to calculate success rate: {e}")
            return 0.0

    @staticmethod
    def get_by_id(log_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific log by its ObjectId.
        
        Args:
            log_id: The string ID of the log.
            
        Returns:
            Dict representing the document, or None if not found.
        """
        if not ObjectId.is_valid(log_id):
            return None
            
        db = SchedulerLog._get_db()
        collection = SchedulerLog._get_collection(db)
        
        try:
            log = collection.find_one({"_id": ObjectId(log_id)})
            if log:
                log["_id"] = str(log["_id"])
            return log
        except Exception as e:
            logger.error(f"Failed to get log by ID {log_id}: {e}")
            return None
        
if __name__ == "__main__":
    import time
    
    print("Testing SchedulerLog Model...")
    try:
        # 1. Create a log
        print("Creating log...")
        log_id = SchedulerLog.create_log(str("manual"))
        print(f"Created log with ID: {log_id}")
        
        # 2. Update the log with a task
        print("Updating log with a task...")
        task = {
            "name": "nasa_refresh",
            "status": "success",
            "startedAt": datetime.now(timezone.utc),
            "completedAt": datetime.now(timezone.utc),
            "recordsUpdated": 150
        }
        SchedulerLog.update_log(log_id, {"$push": {"tasks": task}})
        print("Task added")
        
        # 3. Simulate processing time
        time.sleep(1)
        
        # 4. Complete the log
        print("Completing log...")
        SchedulerLog.complete_log(log_id, status="success", datasets_updated=1, total_datasets=1)
        print("Log completed")
        
        # 5. Get the completed log
        completed_log = SchedulerLog.get_by_id(log_id)
        print(f"Retrieved Log duration: {completed_log.get('duration')} seconds, Status: {completed_log.get('status')}")
        
        # 6. Check Success Rate
        rate = SchedulerLog.get_success_rate()
        print(f"Current 30-day success rate: {rate}%")
        
    except Exception as ex:
        print(f"Test Failed: {ex}")