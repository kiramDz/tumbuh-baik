import os
import logging
import threading
from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify, request
from models.scheduler_logs import SchedulerLog
from middleware.auth_middleware import require_auth

logger = logging.getLogger(__name__)

# Registration Example for app.py:
# from routes.scheduler_routes import scheduler_bp
# app.register_blueprint(scheduler_bp)

scheduler_bp = Blueprint('scheduler', __name__, url_prefix="/api/v1/scheduler")

# Simple in-memory rate limiting tracker for manual triggers
_trigger_history = []

def get_next_run() -> str:
    """Calculate next run from cron expression (daily 02:00 WIB)."""
    # 02:00 WIB is 19:00 UTC previous day.
    now = datetime.now(timezone.utc)
    # Next WIB 02:00:
    # Convert now to WIB (+7)
    now_wib = now + timedelta(hours=7)
    next_wib_run = datetime(now_wib.year, now_wib.month, now_wib.day, 2, 0, 0)
    
    if now_wib >= next_wib_run:
        next_wib_run += timedelta(days=1)
        
    next_utc_run = next_wib_run - timedelta(hours=7)
    return next_utc_run.replace(tzinfo=timezone.utc).isoformat()

def validate_api_key(req) -> bool:
    """Validate API key via Header Authorization."""
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    
    token = auth_header.split(" ")[1]
    expected_token = os.getenv("API_KEY", "dev_secret_key")
    return token == expected_token

def run_scheduler_async(log_id: str, tasks: list):
    """
    Mock async executor for manual triggers.
    In Phase 3, this should ideally invoke the actual daily_scheduler.py logic 
    or trigger the same processes.
    """
    try:
        # Update log to running with targeted tasks
        logger.info(f"Running scheduler async for log: {log_id}. Tasks: {tasks}")
        # Simulaton of async task logic ...
        # After completion:
        # SchedulerLog.complete_log(log_id, status="success")
    except Exception as e:
        logger.error(f"Async scheduler failed: {e}")
        SchedulerLog.complete_log(log_id, status="failed")

@scheduler_bp.route("/status", methods=["GET"])
def get_status():
    """Get Scheduler Status"""
    try:
        latest_log = SchedulerLog.get_latest()
        
        # Calculate statistics
        db = SchedulerLog._get_db()
        collection = db["scheduler_logs"]
        total_execs = collection.count_documents({})
        failed_execs = collection.count_documents({"status": "failed"})
        
        success_rate = SchedulerLog.get_success_rate(days=30)
        
        # Calculate Average Duration
        durations = list(collection.aggregate([
            {"$match": {"status": "success", "duration": {"$ne": None}}},
            {"$group": {"_id": None, "avgDuration": {"$avg": "$duration"}}}
        ]))
        avg_duration = durations[0]["avgDuration"] if durations else 0

        # Check if cron service might active (simple ping check or assume true if enabled in env)
        is_active = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"

        data = {
            "lastRun": latest_log,
            "nextRun": get_next_run(),
            "isActive": is_active,
            "statistics": {
                "successRate": success_rate,
                "avgDuration": round(avg_duration, 2),
                "totalExecutions": total_execs,
                "failedExecutions": failed_execs
            }
        }
        
        return jsonify({"success": True, "data": data}), 200

    except Exception as e:
        logger.error(f"Error fetching scheduler status: {e}")
        return jsonify({
            "success": False, 
            "error": {"code": "INTERNAL_ERROR", "message": "Failed to retrieve status."}
        }), 500

@scheduler_bp.route("/logs", methods=["GET"])
def get_logs():
    """Get Scheduler Logs with Pagination"""
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
        status = request.args.get("status")
        
        # Cap limit
        limit = min(limit, 50)
        
        logs, total = SchedulerLog.get_logs(limit=limit, offset=offset, status_filter=status)
        
        has_more = (offset + limit) < total
        
        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "hasMore": has_more,
                    "nextOffset": offset + limit if has_more else None
                }
            }
        }), 200

    except ValueError as e:
        return jsonify({"success": False, "error": {"code": "VALIDATION_ERROR", "message": str(e)}}), 400
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500

@scheduler_bp.route("/trigger", methods=["POST"])
@require_auth
def trigger_scheduler():
    """Trigger Scheduler Manually"""
    global _trigger_history
    
    # Simple rate limiting: max 5 hits per hour
    now = datetime.now()
    _trigger_history = [t for t in _trigger_history if now - t < timedelta(hours=1)]
    if len(_trigger_history) >= 5:
        return jsonify({"success": False, "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Max 5 manual triggers per hour."}}), 429
        
    # Check if already running
    latest = SchedulerLog.get_latest()
    if latest and latest.get("status") == "running":
         return jsonify({"success": False, "error": {"code": "CONFLICT", "message": "Scheduler is already running."}}), 409
         
    _trigger_history.append(now)

    try:
        body = request.get_json(silent=True) or {}
        tasks = body.get("tasks", [])
        is_async = body.get("async", True)
        
        log_id = SchedulerLog.create_log(triggered_by="manual")
        
        if is_async:
            _thread = threading.Thread(target=run_scheduler_async, args=(log_id, tasks))
            _thread.start()
            
            return jsonify({
                "success": True,
                "data": {
                    "executionId": log_id,
                    "status": "running",
                    "startedAt": datetime.now(timezone.utc).isoformat(),
                    "estimatedDuration": 120
                }
            }), 202
            
        else:
            # Sync execution    
            run_scheduler_async(log_id, tasks)
            # Fetch completed logic
            completed_log = SchedulerLog.get_by_id(log_id)
            return jsonify({
                 "success": True,
                 "data": completed_log
            }), 200
            
    except Exception as e:
        logger.error(f"Trigger error: {e}")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR"}}), 500