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
        db.scheduler_logs.update_one(
            {"_id": ObjectId(log_id)},
            {"$push": {"tasks": task_result}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
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
    for attempt in range(retries):
        try:
            # FIX: Tingkatkan timeout menjadi 3600 detik (1 jam) untuk proses berat
            if method.upper() == 'GET':
                response = http_session.get(url, timeout=3600)
            else:
                response = http_session.post(url, json=payload, timeout=3600)
                
            if response.status_code in [200, 202]:
                return True, response.json()
            elif response.status_code == 401:
                logger.error("Authentication failed. Invalid CRON_SECRET.")
                sys.exit(1)
            elif response.status_code == 400: # Langsung gagalkan jika 400 (Bad Request)
                logger.warning(f"API Error (400): {response.text}")
                return False, {"message": "Bad request or validation error"}
            else:
                logger.warning(f"API Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ReadTimeout as e:
            # FIX: Jika server lama merespons (ReadTimeout), JANGAN di-retry (karena server masih bekerja di background)
            logger.error(f"Timeout on attempt {attempt + 1}. Server might still be processing this heavily in background. Aborting retry to prevent duplicates.")
            return False, {"message": "Server processing timeout."}
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            
        wait_time = 2 * (2 ** attempt)  # 2s, 4s, 8s
        logger.info(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)
        
    return False, {"message": "Max retries reached."}
# Tasks
def task_nasa_refresh(is_dry_run):
    logger.info("Task: Refreshing NASA data...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing NASA refresh")
        return {"name": "nasa_refresh", "status": "success"}

    # NASA refresh-all ada di Next.js (Hono backend)
    success, data = make_api_call("POST", NEXT_API_BASE_URL, "/api/v1/nasa-power/refresh-all")
    if success:
        logger.info(f"SUCCESS: NASA refresh completed.")
        return {"name": "nasa_refresh", "status": "success", "response": data}

    else:
        logger.error("FAILED: NASA refresh failed.")
        return {"name": "nasa_refresh", "status": "failed", "error": data}

def task_bmkg_preprocess(is_dry_run):
    logger.info("Task: Preprocessing BMKG datasets...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing BMKG preprocess")
        return {"name": "bmkg_preprocess", "status": "success"}
    
    # 1. Fetch BMKG datasets (metadata ada di Next.js Hono routes)
    success, data = make_api_call("GET", NEXT_API_BASE_URL, "/api/v1/dataset-meta")
    if not success:
        return {"name": "bmkg_preprocess", "status": "failed", "error": "Could not fetch datasets list"}
    
    # Data dari Next.js biasanya ada di dalam format data.data jika di-wrap Response
    datasets = data if isinstance(data, list) else data.get("data", [])
    
    # FIX: Filter dataset yang HANYA berstatus "raw" dan mengandung kata "bmkg" di source/dataType
    bmkg_datasets = []
    for d in datasets:
        # Pengecekan apakah ini data BMKG (cek kata "bmkg" ada di dalam source atau dataType)
        is_bmkg = (
            "bmkg" in str(d.get("source", "")).lower() or 
            "bmkg" in str(d.get("dataType", "")).lower()
        )
        # Pengecekan apakah statusnya masih raw (belum dipreprocess)
        is_raw = str(d.get("status", "")).lower() == "raw"
        
        if is_bmkg and is_raw:
            bmkg_datasets.append(d["collectionName"])
    
    if not bmkg_datasets:
        logger.info("No RAW BMKG datasets found to process.")
        return {"name": "bmkg_preprocess", "status": "success", "processed": 0}
        
    logger.info(f"Found {len(bmkg_datasets)} RAW BMKG datasets. Starting preprocessing loop...")
    count_success = 0
    errors = []
    
    for name in bmkg_datasets:
        logger.info(f" -> Processing {name}")
        # Preprocessing eksekusi aslinya ada di Flask
        c_success, c_data = make_api_call("POST", FLASK_API_BASE_URL, f"/api/v1/preprocess/bmkg/{name}")
        if c_success:
            count_success += 1
        else:
            errors.append(f"Failed on {name}")

    if count_success == len(bmkg_datasets):
        logger.info(f"SUCCESS: BMKG preprocessing completed ({count_success}/{len(bmkg_datasets)})")
        return {"name": "bmkg_preprocess", "status": "success"}
    else:
        logger.warning(f"PARTIAL: BMKG preprocessing ({count_success}/{len(bmkg_datasets)})")
        return {"name": "bmkg_preprocess", "status": "failed" if count_success == 0 else "partial"}

def task_nasa_preprocess(is_dry_run):
    logger.info("Task: Preprocessing NASA datasets...")
    if is_dry_run:
        logger.info("DRY RUN: Bypassing NASA preprocess")
        return {"name": "nasa_preprocess", "status": "success"}

    success, data = make_api_call("GET", NEXT_API_BASE_URL, "/api/v1/dataset-meta")
    if not success:
        return {"name": "nasa_preprocess", "status": "failed", "error": "Could not fetch datasets list"}

    datasets = data if isinstance(data, list) else data.get("data", [])
    
    nasa_datasets = []
    for d in datasets:
        # Pengecekan apakah ini data NASA
        is_nasa = (
            "nasa" in str(d.get("source", "")).lower() or 
            "nasa" in str(d.get("dataType", "")).lower()
        )
        # Pengecekan apakah statusnya raw atau latest
        status = str(d.get("status", "")).lower()
        is_pending_preprocess = status in ["raw", "latest"]
        
        if is_nasa and is_pending_preprocess:
            nasa_datasets.append(d["collectionName"])

    if not nasa_datasets:
        logger.info("No pending NASA datasets found to process.")
        return {"name": "nasa_preprocess", "status": "success", "processed": 0}

    logger.info(f"Found {len(nasa_datasets)} pending NASA datasets. Starting preprocessing loop...")
    count_success = 0
    errors = []

    for name in nasa_datasets:
        logger.info(f" -> Processing {name}")
        c_success, _ = make_api_call("POST", FLASK_API_BASE_URL, f"/api/v1/preprocess/nasa/{name}")
        if c_success:
            count_success += 1
        else:
            errors.append(f"Failed on {name}")

    if count_success == len(nasa_datasets):
        logger.info(f"SUCCESS: NASA preprocessing completed ({count_success}/{len(nasa_datasets)})")
        return {"name": "nasa_preprocess", "status": "success"}
    else:
        logger.warning(f"PARTIAL: NASA preprocessing ({count_success}/{len(nasa_datasets)})")
        return {"name": "nasa_preprocess", "status": "failed" if count_success == 0 else "partial"}

def main():
    parser = argparse.ArgumentParser(description="Climate Automation Scheduler")
    parser.add_argument("--manual", action="store_true", help="Mark execution as manual")
    # FIX: Tambahkan nasa_preprocess ke choices 
    parser.add_argument("--tasks", choices=["nasa_refresh", "nasa_preprocess", "bmkg_preprocess", "all"], default="all", help="Specific tasks to run")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without side effects")
    args = parser.parse_args()

    trigger_type = "manual" if args.manual else "cron"
    if args.dry_run:
        logger.info("=== STARTING SCHEDULER IN DRY-RUN MODE ===")
        log_id = None
    else:
        logger.info("=== STARTING SCHEDULER EXECUTION ===")
        log_id = create_log(trigger_type)
        if not log_id: sys.exit(1)

    start_time = time.time()
    results = []
    
    # Execution (Sekarang berurutan: Refresh NASA -> Preprocess NASA -> Preprocess BMKG)
    if args.tasks in ["nasa_refresh", "all"]:
        res_nasa = task_nasa_refresh(args.dry_run)
        results.append(res_nasa)
        update_task_log(log_id, res_nasa)

    # TAMBAHKAN PEMANGGILAN NASA PREPROCESS DISINI
    if args.tasks in ["nasa_preprocess", "all"]:
        res_nasa_pre = task_nasa_preprocess(args.dry_run)
        results.append(res_nasa_pre)
        update_task_log(log_id, res_nasa_pre)

    if args.tasks in ["bmkg_preprocess", "all"]:
        res_bmkg = task_bmkg_preprocess(args.dry_run)
        results.append(res_bmkg)
        update_task_log(log_id, res_bmkg)

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