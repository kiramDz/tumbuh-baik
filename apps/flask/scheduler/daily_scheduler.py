import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

# Load env 
load_dotenv()

# Constants
FLASK_API_BASE_URL = os.getenv("FLASK_API_BASE_URL")
NEXT_API_BASE_URL = os.getenv("NEXT_API_BASE_URL")
CRON_SECRET = os.getenv("CRON_SECRET")
MONGO_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Request session for connection pooling
http_session = requests.Session()
http_session.headers.update({
    "Content-Type": "application/json",
    "X-Cron-Secret": CRON_SECRET
})

def setup_logger():
    """Setup dual output logging (Console & File)."""
    logger = logging.getLogger("ClimateScheduler")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    try:
        log_file = "/var/log/climate_scheduler.log"
        if not os.path.exists(os.path.dirname(log_file)):
            log_file = "climate_scheduler.log" # Fallback to local
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"Could not setup file logger: {e}. Using console only.")
        
    return logger

logger = setup_logger()

# MongoDB Helpers (Direct manipulation for atomicity without relying on Flask model file path issues)
def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGODB_DB_NAME]

def create_log(triggered_by="cron"):
    try:
        db = get_db()
        log_doc = {
            "executedAt": datetime.now(timezone.utc),
            "status": "running",
            "triggeredBy": triggered_by,
            "tasks": [],
            "totalDatasets": 0,
            "datasetsUpdated": 0,
            "errors": [],
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc)
        }
        res = db.scheduler_logs.insert_one(log_doc)
        return str(res.inserted_id)
    except Exception as e:
        logger.error(f"Failed to create start log info in MongoDB: {e}")
        return None

def update_task_log(log_id, task_result):
    if not log_id: return
    try:
        db = get_db()
        from bson import ObjectId
        
        update_op = {
            "$push": {"tasks": task_result}, 
            "$set": {"updatedAt": datetime.now(timezone.utc)}
        }
        
        # Safely push specific errors directly into the document's root errors array
        if task_result.get("errors"):
            update_op["$push"]["errors"] = {"$each": task_result["errors"]}
            
        db.scheduler_logs.update_one(
            {"_id": ObjectId(log_id)},
            update_op
        )
    except Exception as e:
        logger.error(f"Failed to update MongoDB log for task: {e}")

def finish_log(log_id, status, datasets_updated=0, total_datasets=0):
    if not log_id: return
    try:
        db = get_db()
        from bson import ObjectId
        now = datetime.now(timezone.utc)
        
        log = db.scheduler_logs.find_one({"_id": ObjectId(log_id)})
        
        # FIX: Tangani perbedaan naive dan aware datetime
        duration = 0
        if log and log.get("executedAt"):
            executed_at = log["executedAt"]
            if executed_at.tzinfo is None:
                executed_at = executed_at.replace(tzinfo=timezone.utc)
            duration = max(0, (now - executed_at).total_seconds())
        
        db.scheduler_logs.update_one(
            {"_id": ObjectId(log_id)},
            {"$set": {
                "status": status,
                "completedAt": now,
                "duration": duration,
                "datasetsUpdated": datasets_updated,
                "totalDatasets": total_datasets,
                "updatedAt": now
            }}
        )
    except Exception as e:
        logger.error(f"Failed to finish MongoDB log: {e}")

# API Wrapper
def make_api_call(method, base_url, endpoint, payload=None, retries=3):
    """Make API call with retry mechanism and exponential backoff."""
    url = f"{base_url}{endpoint}"
    last_error = {"message": "Max retries reached without a specific error."}
    
    for attempt in range(retries):
        try:
            # FIX: Tingkatkan timeout menjadi 3600 detik (1 jam) untuk proses berat
            if method.upper() == 'GET':
                response = http_session.get(url, timeout=3600)
            else:
                response = http_session.post(url, json=payload, timeout=3600)
                
            if response.status_code in [200, 202]:
                try:
                    return True, response.json()
                except ValueError:
                    return True, {"message": "Success", "text": response.text}
                    
            elif response.status_code == 401:
                logger.error("Authentication failed. Invalid CRON_SECRET.")
                sys.exit(1)
            else:
                # Safely parse actual error response instead of generic strings
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", err_json.get("message", response.text))
                    # Handle if 'error' is a dict (like your Flask internal errors)
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", str(err_msg))
                except ValueError:
                    err_msg = response.text if response.text else "Empty response body"
                
                last_error = {"message": f"HTTP {response.status_code}: {err_msg}"}
                
                if response.status_code == 400: # Langsung gagalkan jika 400 (Bad Request)
                    logger.warning(f"API Error (400) Unrecoverable: {err_msg}")
                    return False, last_error
                else:
                    logger.warning(f"API Error ({response.status_code}) on attempt {attempt + 1}: {err_msg}")
                
        except requests.exceptions.ReadTimeout as e:
            # FIX: Jangan langsung gagal, biarkan retry bekerja (catat di last_error)
            last_error = {"message": f"Server processing timeout (ReadTimeout) on attempt {attempt + 1}."}
            logger.warning(f"Timeout on attempt {attempt + 1}. The server might still be working, but we will retry to ensure completion...")
            
        except requests.exceptions.RequestException as e:
            last_error = {"message": f"Connection Error: {str(e)}"}
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            
        # Only sleep if it's not the last attempt
        if attempt < retries - 1:
            wait_time = 2 * (2 ** attempt)  # 2s, 4s, 8s
            logger.info(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
        
    return False, last_error


# Tasks
def task_nasa_refresh(is_dry_run, target_datasets=None):
    logger.info("Task: Refreshing NASA data...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing NASA refresh")
        return {"name": "nasa_refresh", "status": "success", "errors": []}

    errors = []
    if target_datasets:
        logger.info(f"Custom datasets requested for refresh: {len(target_datasets)} datasets. Processing individually...")
        count_success = 0
        for name in target_datasets:
            logger.info(f" -> Refreshing {name}")
            success, resp_data = make_api_call("POST", NEXT_API_BASE_URL, f"/api/v1/nasa-power/refresh/{name}")
            if success: 
                count_success += 1
            else:
                error_msg = resp_data.get("message", "Unknown error") if isinstance(resp_data, dict) else str(resp_data)
                logger.error(f"   FAILED: {name} - {error_msg}")
                errors.append(f"Refresh {name}: {error_msg}")
            
        final_status = "success" if count_success == len(target_datasets) else "partial" if count_success > 0 else "failed"
        return {"name": "nasa_refresh", "status": final_status, "errors": errors}
    else:
        # Quick run: Eksekusi semua
        logger.info("Triggering generic refresh-all...")
        success, resp_data = make_api_call("POST", NEXT_API_BASE_URL, "/api/v1/nasa-power/refresh-all")
        if success:
            logger.info(f"SUCCESS: NASA refresh completed.")
            return {"name": "nasa_refresh", "status": "success", "errors": []}
        else:
            logger.error("FAILED: NASA refresh failed.")
            error_msg = resp_data.get("error", resp_data.get("message", "Unknown error")) if isinstance(resp_data, dict) else str(resp_data)
            return {"name": "nasa_refresh", "status": "failed", "errors": [f"Refresh All: {error_msg}"]}
        

def task_bmkg_preprocess(is_dry_run, target_datasets=None):
    logger.info("Task: Preprocessing BMKG datasets...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing BMKG preprocess")
        return {"name": "bmkg_preprocess", "status": "success", "errors": []}
    
    if target_datasets:
        bmkg_datasets = target_datasets
    else:
        # Fetch auto dari list raw
        success, data = make_api_call("GET", NEXT_API_BASE_URL, "/api/v1/dataset-meta")
        if not success:
            return {"name": "bmkg_preprocess", "status": "failed", "errors": ["Failed to fetch dataset meta"]}
        
        datasets = data if isinstance(data, list) else data.get("data", [])
        bmkg_datasets = [
            d["collectionName"] for d in datasets 
            if ("bmkg" in str(d.get("source", "")).lower() or "bmkg" in str(d.get("dataType", "")).lower()) 
            and str(d.get("status", "")).lower() == "raw"
        ]
    
    if not bmkg_datasets:
        logger.info("No RAW BMKG datasets found to process.")
        return {"name": "bmkg_preprocess", "status": "success", "processed": 0, "errors": []}
        
    logger.info(f"Found {len(bmkg_datasets)} BMKG datasets. Starting preprocessing...")
    count_success = 0
    errors = []
    
    for name in bmkg_datasets:
        logger.info(f" -> Processing {name}")
        c_success, resp_data = make_api_call("POST", FLASK_API_BASE_URL, f"/api/v1/preprocess/bmkg/{name}")
        if c_success: 
            count_success += 1
        else:
            error_msg = resp_data.get("message", "Unknown error") if isinstance(resp_data, dict) else str(resp_data)
            logger.error(f"   FAILED: {name} - {error_msg}")
            errors.append(f"Preprocess BMKG {name}: {error_msg}")

    final_status = "success" if count_success == len(bmkg_datasets) else "partial" if count_success > 0 else "failed"
    return {"name": "bmkg_preprocess", "status": final_status, "errors": errors}

def task_nasa_preprocess(is_dry_run, target_datasets=None):
    logger.info("Task: Preprocessing NASA datasets...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing NASA preprocess")
        return {"name": "nasa_preprocess", "status": "success", "errors": []}

    if target_datasets:
        nasa_datasets = target_datasets
    else:
        success, data = make_api_call("GET", NEXT_API_BASE_URL, "/api/v1/dataset-meta")
        if not success: return {"name": "nasa_preprocess", "status": "failed", "errors": ["Failed to fetch dataset meta"]}

        datasets = data if isinstance(data, list) else data.get("data", [])
        nasa_datasets = [
            d["collectionName"] for d in datasets 
            if ("nasa" in str(d.get("source", "")).lower() or "nasa" in str(d.get("dataType", "")).lower()) 
            and str(d.get("status", "")).lower() in ["raw", "latest"]
        ]

    if not nasa_datasets:
        logger.info("No pending NASA datasets found.")
        return {"name": "nasa_preprocess", "status": "success", "processed": 0, "errors": []}

    logger.info(f"Found {len(nasa_datasets)} NASA datasets. Starting...")
    count_success = 0
    errors = []

    for name in nasa_datasets:
        logger.info(f" -> Processing {name}")
        c_success, resp_data = make_api_call("POST", FLASK_API_BASE_URL, f"/api/v1/preprocess/nasa/{name}")
        if c_success: 
            count_success += 1
        else:
            error_msg = resp_data.get("message", "Unknown error") if isinstance(resp_data, dict) else str(resp_data)
            logger.error(f"   FAILED: {name} - {error_msg}")
            errors.append(f"Preprocess NASA {name}: {error_msg}")

    final_status = "success" if count_success == len(nasa_datasets) else "partial" if count_success > 0 else "failed"
    return {"name": "nasa_preprocess", "status": final_status, "errors": errors}

def main():
    parser = argparse.ArgumentParser(description="Climate Automation Scheduler")
    parser.add_argument("--manual", action="store_true", help="Mark execution as manual")
    parser.add_argument("--tasks", choices=["nasa_refresh", "nasa_preprocess", "bmkg_preprocess", "all"], default="all", help="Specific tasks to run")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without side effects")
    parser.add_argument("--log-id", help="Existing MongoDB log ID to update")
    
    # Custom Selection Arguments
    parser.add_argument("--nasa-refresh", help="Comma-separated NASA datasets to refresh")
    parser.add_argument("--nasa-preprocess", help="Comma-separated NASA datasets to preprocess")
    parser.add_argument("--bmkg-preprocess", help="Comma-separated BMKG datasets to preprocess")
    args = parser.parse_args()

    trigger_type = "manual" if args.manual else "cron"
    
    # Gunakan log_id dari argumen jika ada, jika tidak buat baru
    if args.log_id:
        log_id = args.log_id
    else:
        log_id = None if args.dry_run else create_log(trigger_type)
        
    if not args.dry_run and not log_id: sys.exit(1)

    start_time = time.time()
    results = []
    
    # Parse custom lists
    t_nasa_refresh = args.nasa_refresh.split(",") if args.nasa_refresh else None
    t_nasa_pre = args.nasa_preprocess.split(",") if args.nasa_preprocess else None
    t_bmkg_pre = args.bmkg_preprocess.split(",") if args.bmkg_preprocess else None

    # Logic: Jika argumen spesifik (custom) diberikan, kita jalankan task tersebut tanpa peduli --tasks
    is_custom = bool(t_nasa_refresh or t_nasa_pre or t_bmkg_pre)

    if is_custom:
        logger.info("=== STARTING SCHEDULER (CUSTOM SELECTION) ===")
        if t_nasa_refresh:
            res = task_nasa_refresh(args.dry_run, t_nasa_refresh)
            results.append(res); update_task_log(log_id, res)
            if t_nasa_pre or t_bmkg_pre:
                logger.info("Cooling down server resources for 15s...")
                time.sleep(15)
                
        if t_nasa_pre:
            res = task_nasa_preprocess(args.dry_run, t_nasa_pre)
            results.append(res); update_task_log(log_id, res)
            if t_bmkg_pre:
                logger.info("Cooling down server resources for 15s...")
                time.sleep(15)
                
        if t_bmkg_pre:
            res = task_bmkg_preprocess(args.dry_run, t_bmkg_pre)
            results.append(res); update_task_log(log_id, res)
    else:
        logger.info("=== STARTING SCHEDULER (QUICK RUN) ===")
        if args.tasks in ["nasa_refresh", "all"]:
            res = task_nasa_refresh(args.dry_run)
            results.append(res); update_task_log(log_id, res)
            if args.tasks == "all":
                logger.info("Cooling down server resources for 15s...")
                time.sleep(15)
                
        if args.tasks in ["nasa_preprocess", "all"]:
            res = task_nasa_preprocess(args.dry_run)
            results.append(res); update_task_log(log_id, res)
            if args.tasks == "all":
                logger.info("Cooling down server resources for 15s...")
                time.sleep(15)
                
        if args.tasks in ["bmkg_preprocess", "all"]:
            res = task_bmkg_preprocess(args.dry_run)
            results.append(res); update_task_log(log_id, res)

    # Summarize Final Status
    statuses = [r["status"] for r in results]
    if all(s == "success" for s in statuses):
        final_status = "success"
        exit_code = 0
    elif all(s == "failed" for s in statuses):
        final_status = "failed"
        exit_code = 1
    else:
        final_status = "partial"
        exit_code = 2

    # Finish log
    if not args.dry_run:
        finish_log(log_id, final_status, total_datasets=len(results), datasets_updated=statuses.count("success"))

    duration = round(time.time() - start_time, 2)
    logger.info(f"=== SCHEDULER COMPLETED [{final_status.upper()}] IN {duration}s ===")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()