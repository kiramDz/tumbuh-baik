from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import calendar

def generate_monthly_summary(config_id, client=None):
    """
    Generate monthly planting calendar summary from daily forecast data
    """
    if client is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        MONGO_URI = os.getenv("MONGODB_URI")
        client = MongoClient(MONGO_URI)
        should_close_client = True
    else:
        should_close_client = False
    
    db = client["tugas_akhir"]
    
    try:
        print(f"[INFO] Generating monthly summary for config_id: {config_id}")
        
        # Fetch forecast data dari holt-winter collection
        forecast_data = list(db["holt-winter"].find({"config_id": config_id}).sort("forecast_date", 1))
        
        if not forecast_data:
            print(f"[WARNING] No forecast data found for config_id: {config_id}")
            return {"error": "No forecast data found"}
        
        print(f"[INFO] Found {len(forecast_data)} forecast records")
        
        # Convert to DataFrame untuk easy grouping
        df_data = []
        for record in forecast_data:
            forecast_date = record["forecast_date"]
            parameters = record.get("parameters", {})
            
            # Extract parameter values
            row = {
                "forecast_date": forecast_date,
                "month": forecast_date.strftime("%Y-%m"),
                "config_id": config_id
            }
            
            # Dynamic parameter extraction
            for param_name, param_data in parameters.items():
                if isinstance(param_data, dict) and "forecast_value" in param_data:
                    row[param_name] = param_data["forecast_value"]
                else:
                    row[param_name] = param_data
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Group by month and calculate statistics
        monthly_summaries = []
        
        for month, month_data in df.groupby("month"):
            print(f"[INFO] Processing month: {month}")
            
            # Calculate rainfall statistics (main parameter for now)
            rainfall_data = month_data.get("RR_imputed", pd.Series())
            
            if rainfall_data.empty:
                print(f"[WARNING] No RR_imputed data for month {month}")
                continue
            
            rainfall_avg = rainfall_data.mean()
            rainfall_total = rainfall_data.sum()
            
            # Determine KT period
            kt_period = determine_kt_period(month)
            
            # Determine planting status
            status, reason, shift_analysis = determine_planting_status(
                month, rainfall_avg, rainfall_total, kt_period
            )
            
            # Build parameters object dynamically
            parameters = {}
            
            # Add RR_imputed stats
            if not rainfall_data.empty:
                parameters["RR_imputed"] = {
                    "avg": round(rainfall_avg, 2),
                    "total": round(rainfall_total, 2),
                    "threshold_status": get_threshold_status(rainfall_avg, status)
                }
            
            # Add other parameters if exist
            for col in df.columns:
                if col not in ["forecast_date", "month", "config_id", "RR_imputed"]:
                    param_data = month_data[col]
                    if not param_data.empty:
                        parameters[col] = {
                            "avg": round(param_data.mean(), 2),
                            "total": round(param_data.sum(), 2)
                        }
            
            # Create summary document
            summary_doc = {
                "month": month,
                "kt_period": kt_period,
                "status": status,
                "shift_analysis": shift_analysis,
                "rainfall_avg": round(rainfall_avg, 2) if not rainfall_data.empty else 0,
                "rainfall_total": round(rainfall_total, 2) if not rainfall_data.empty else 0,
                "reason": reason,
                "parameters": parameters,
                "config_id": config_id,
                "created_at": datetime.now().isoformat(),
                "location": "aceh_besar"
            }
            
            monthly_summaries.append(summary_doc)
        
        # Save to holt-winter-summary collection
        if monthly_summaries:
            # Clear existing summaries for this config
            db["holt-winter-summary"].delete_many({"config_id": config_id})
            
            # Insert new summaries
            db["holt-winter-summary"].insert_many(monthly_summaries)
            
            print(f"✓ Generated {len(monthly_summaries)} monthly summaries")
            return {
                "success": True,
                "summaries_generated": len(monthly_summaries),
                "months_processed": [doc["month"] for doc in monthly_summaries]
            }
        else:
            print("[WARNING] No monthly summaries generated")
            return {"error": "No monthly summaries generated"}
    
    except Exception as e:
        print(f"[ERROR] Failed to generate monthly summary: {str(e)}")
        return {"error": str(e)}
    
    finally:
        if should_close_client:
            client.close()

def determine_kt_period(month_str):
    """
    Determine KT period based on month
    KT-1: Sept-Jan, KT-2: Feb-Jun, KT-3: Jul-Aug
    """
    month_num = int(month_str.split("-")[1])
    
    if month_num in [9, 10, 11, 12, 1]:  # Sept-Jan
        return "KT-1"
    elif month_num in [2, 3, 4, 5, 6]:  # Feb-Jun
        return "KT-2"
    else:  # Jul-Aug
        return "KT-3"

def determine_planting_status(month, rainfall_avg, rainfall_total, kt_period):
    """
    Determine planting status based on rainfall and KT period
    Thresholds for Aceh Besar (estimated)
    """
    month_num = int(month.split("-")[1])
    
    # Threshold curah hujan untuk Aceh Besar (mm/bulan)
    THRESHOLDS = {
        "tanam_min": 200,      
        "tanam_optimal": 300,  
        "panen_max": 100       
    }
    
    
    if kt_period == "KT-1": 
        if month_num in [9, 10]: 
            if rainfall_avg >= THRESHOLDS["tanam_optimal"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, optimal untuk tanam (≥{THRESHOLDS['tanam_optimal']}mm)", "normal"
            elif rainfall_avg >= THRESHOLDS["tanam_min"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, cukup untuk tanam (≥{THRESHOLDS['tanam_min']}mm)", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, di bawah threshold tanam ({THRESHOLDS['tanam_min']}mm)", "kekeringan"
        elif month_num in [11, 12]:  
            if rainfall_avg >= THRESHOLDS["tanam_min"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, mendukung pertumbuhan", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, tidak mendukung pertumbuhan", "kekeringan"
        else:  # Jan: Periode panen
            if rainfall_avg <= THRESHOLDS["panen_max"]:
                return "panen", f"Curah hujan {rainfall_avg}mm/bulan, cocok untuk panen (≤{THRESHOLDS['panen_max']}mm)", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, terlalu tinggi untuk panen", "hujan berlebih"
    
    elif kt_period == "KT-2":  
        if month_num in [2, 3]:  
            if rainfall_avg >= THRESHOLDS["tanam_optimal"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, optimal untuk tanam (≥{THRESHOLDS['tanam_optimal']}mm)", "normal"
            elif rainfall_avg >= THRESHOLDS["tanam_min"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, cukup untuk tanam (≥{THRESHOLDS['tanam_min']}mm)", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, di bawah threshold tanam ({THRESHOLDS['tanam_min']}mm)", "kekeringan"
        elif month_num in [4, 5]:  # Apr-May: Periode pertumbuhan
            if rainfall_avg >= THRESHOLDS["tanam_min"]:
                return "tanam", f"Curah hujan {rainfall_avg}mm/bulan, mendukung pertumbuhan", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, tidak mendukung pertumbuhan", "kekeringan"
        else:  # Jun: Periode panen
            if rainfall_avg <= THRESHOLDS["panen_max"]:
                return "panen", f"Curah hujan {rainfall_avg}mm/bulan, cocok untuk panen (≤{THRESHOLDS['panen_max']}mm)", "normal"
            else:
                return "tidak cocok tanam", f"Curah hujan {rainfall_avg}mm/bulan, terlalu tinggi untuk panen", "hujan berlebih"
    
    else:  # KT-3: Jul-Aug (Musim Istirahat)
        return "rehat", f"Periode istirahat, curah hujan {rainfall_avg}mm/bulan", "normal"

def get_threshold_status(rainfall_avg, status):
    """
    Get threshold status based on rainfall and planting status
    """
    if status == "tanam":
        if rainfall_avg >= 300:
            return "optimal_planting_threshold"
        elif rainfall_avg >= 200:
            return "minimum_planting_threshold"
        else:
            return "below_planting_threshold"
    elif status == "panen":
        if rainfall_avg <= 100:
            return "optimal_harvest_threshold"
        else:
            return "above_harvest_threshold"
    elif status == "rehat":
        return "resting_period"
    else:  # tidak cocok tanam
        return "unsuitable_threshold"

# Modifikasi untuk ditambahkan ke run_forecast_from_config()
def add_to_forecast_config():
    """
    Tambahkan ini ke akhir function run_forecast_from_config() setelah:
    db.forecast_configs.update_one({"_id": config["_id"]}, {"$set": {"status": "done"}})
    """
    
    # Trigger monthly summary generation
    print("[INFO] Generating monthly planting calendar summary...")
    summary_result = generate_monthly_summary(config_id, client)
    
    if summary_result.get("success"):
        print(f"✓ Monthly summary generated: {summary_result['summaries_generated']} months")
        
        # Update config with summary status
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {
                "status": "done",
                "summary_generated": True,
                "summary_months": summary_result["summaries_generated"],
                "summary_collection": "holt-winter-summary"
            }}
        )
    else:
        print(f"✗ Monthly summary generation failed: {summary_result.get('error', 'Unknown error')}")
        
        # Update config with error status
        db.forecast_configs.update_one(
            {"_id": config["_id"]},
            {"$set": {
                "status": "done",
                "summary_generated": False,
                "summary_error": summary_result.get("error", "Unknown error")
            }}
        )
    
    return jsonify({
        "message": f"Forecasting completed for config: {name}",
        "forecastResultCollection": forecast_coll,
        "results": convert_objectid(results),
        "total_forecast_dates": len(forecast_data),
        "summary_result": summary_result
    }), 200