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
        
        # Evaluate KT periods first based on first month rainfall
        kt_evaluations = evaluate_kt_periods(df)
        
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
            
            # Get KT evaluation result
            kt_suitable = bool(kt_evaluations.get(kt_period, {}).get("suitable", False))  # Convert to native Python bool
            kt_reason = str(kt_evaluations.get(kt_period, {}).get("reason", ""))  # Ensure string type
            
            # Determine planting status
            status, reason, shift_analysis = determine_planting_status(
                month, rainfall_avg, rainfall_total, kt_period, kt_suitable, kt_reason
            )
            
            # Build parameters object dynamically
            parameters = {}
            
            # Add RR_imputed stats
            if not rainfall_data.empty:
                parameters["RR_imputed"] = {
                    "avg": round(float(rainfall_avg), 2),  # Convert to native Python float
                    "total": round(float(rainfall_total), 2),  # Convert to native Python float
                    "threshold_status": get_threshold_status(rainfall_avg, status, kt_suitable)
                }
            
            # Add other parameters if exist
            for col in df.columns:
                if col not in ["forecast_date", "month", "config_id", "RR_imputed"]:
                    param_data = month_data[col]
                    if not param_data.empty:
                        parameters[col] = {
                            "avg": round(float(param_data.mean()), 2),  # Convert to native Python float
                            "total": round(float(param_data.sum()), 2)  # Convert to native Python float
                        }
            
            # Create summary document
            summary_doc = {
                "month": str(month),  # Ensure string type
                "kt_period": str(kt_period),  # Ensure string type
                "kt_suitable": kt_suitable,  # Already converted to native Python bool
                "kt_evaluation": kt_evaluations.get(kt_period, {}),
                "status": str(status),  # Ensure string type
                "shift_analysis": str(shift_analysis),  # Ensure string type
                "rainfall_avg": round(float(rainfall_avg), 2) if not rainfall_data.empty else 0.0,
                "rainfall_total": round(float(rainfall_total), 2) if not rainfall_data.empty else 0.0,
                "reason": str(reason),  # Ensure string type
                "parameters": parameters,
                "config_id": str(config_id),  # Ensure string type
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
                "months_processed": [doc["month"] for doc in monthly_summaries],
                "kt_evaluations": kt_evaluations
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

def evaluate_kt_periods(df):
    """
    Evaluate each KT period based on first month rainfall total
    KT suitable if first month has ≥50mm total rainfall
    """
    kt_evaluations = {}
    
    # Define first months for each KT period
    kt_first_months = {
        "KT-1": 9,   # September
        "KT-2": 2,   # February  
        "KT-3": 7    # July
    }
    
    for kt_period, first_month_num in kt_first_months.items():
        # Find data for the first month of this KT period
        first_month_data = df[df["month"].str.endswith(f"-{first_month_num:02d}")]
        
        if not first_month_data.empty:
            # Calculate total rainfall for first month
            rainfall_data = first_month_data.get("RR_imputed", pd.Series())
            
            if not rainfall_data.empty:
                rainfall_total = float(rainfall_data.sum())  # Convert to native Python float
                
                # Determine suitability based on 50mm threshold
                suitable = bool(rainfall_total >= 50)  # Convert to native Python bool
                
                kt_evaluations[kt_period] = {
                    "suitable": suitable,
                    "first_month_rainfall": round(rainfall_total, 2),
                    "first_month": int(first_month_num),  # Ensure it's native Python int
                    "threshold": 50,
                    "reason": f"Bulan pertama {kt_period} ({calendar.month_name[first_month_num]}) memiliki curah hujan total {rainfall_total:.1f}mm {'≥' if suitable else '<'} 50mm"
                }
            else:
                kt_evaluations[kt_period] = {
                    "suitable": False,
                    "first_month_rainfall": 0.0,
                    "first_month": int(first_month_num),
                    "threshold": 50,
                    "reason": f"Tidak ada data curah hujan untuk bulan pertama {kt_period}"
                }
        else:
            kt_evaluations[kt_period] = {
                "suitable": False,
                "first_month_rainfall": 0.0,
                "first_month": int(first_month_num),
                "threshold": 50,
                "reason": f"Tidak ada data untuk bulan pertama {kt_period}"
            }
    
    return kt_evaluations

def determine_kt_period(month_str):
    """
    Determine KT period based on month with flexible ranges
    KT-1: Late Sept-Jan OR Early Oct-Feb (4-5 bulan)
    KT-2: Late Feb-Jun OR Early Mar-Jun (4 bulan)
    KT-3: Late Jun-Oct OR Early Jul-Oct (4 bulan)
    """
    month_num = int(month_str.split("-")[1])
    
    # KT-1: September sampai Januari (bisa diperpanjang sampai Februari)
    if month_num in [9, 10, 11, 12, 1, 2]:
        return "KT-1"
    # KT-2: Februari sampai Juni (bisa mulai awal Maret)
    elif month_num in [2, 3, 4, 5, 6]:
        return "KT-2"
    # KT-3: Juni sampai Oktober (biasanya Juli-September/Oktober)
    elif month_num in [6, 7, 8, 9, 10]:
        return "KT-3"
    else:
        return "Unknown KT"  # Fallback untuk bulan di luar range

def determine_planting_status(month, rainfall_avg, rainfall_total, kt_period, kt_suitable, kt_reason):
    """
    Determine planting status based on KT suitability and month position
    Status tetap ditentukan per bulan untuk keterangan, tapi mengacu pada evaluasi KT
    """
    month_num = int(month.split("-")[1])
    
    # Jika KT tidak suitable, semua bulan dalam KT tersebut tidak cocok tanam
    if not kt_suitable:
        if kt_period == "KT-3":
            return "rehat", f"Periode istirahat. {kt_reason}", "kt_not_suitable"
        else:
            return "tidak cocok tanam", f"KT tidak cocok tanam. {kt_reason}", "kt_not_suitable"
    
    # Jika KT suitable, tentukan status berdasarkan posisi bulan dalam siklus tanam
    if kt_period == "KT-1":
        if month_num in [9, 10]:  # Bulan tanam
            return "tanam", f"Bulan tanam KT-1. Curah hujan {rainfall_avg:.1f}mm/bulan mendukung penanaman", "normal"
        elif month_num in [11, 12]:  # Bulan pertumbuhan
            return "tanam", f"Bulan pertumbuhan KT-1. Curah hujan {rainfall_avg:.1f}mm/bulan mendukung pertumbuhan tanaman", "normal"
        else:  # Jan: Bulan panen
            return "panen", f"Bulan panen KT-1. Curah hujan {rainfall_avg:.1f}mm/bulan", "normal"
    
    elif kt_period == "KT-2":
        if month_num in [2, 3]:  # Bulan tanam
            return "tanam", f"Bulan tanam KT-2. Curah hujan {rainfall_avg:.1f}mm/bulan mendukung penanaman", "normal"
        elif month_num in [4, 5]:  # Bulan pertumbuhan
            return "tanam", f"Bulan pertumbuhan KT-2. Curah hujan {rainfall_avg:.1f}mm/bulan mendukung pertumbuhan tanaman", "normal"
        else:  # Jun: Bulan panen
            return "panen", f"Bulan panen KT-2. Curah hujan {rainfall_avg:.1f}mm/bulan", "normal"
    
    else:  # KT-3: Jul-Aug (Musim Istirahat)
        return "rehat", f"Periode istirahat KT-3. Curah hujan {rainfall_avg:.1f}mm/bulan", "normal"

def get_threshold_status(rainfall_avg, status, kt_suitable):
    """
    Get threshold status based on rainfall, planting status, and KT suitability
    """
    if not kt_suitable:
        return "kt_not_suitable"
    
    if status == "tanam":
        return "kt_suitable_planting"
    elif status == "panen":
        return "kt_suitable_harvest"
    elif status == "rehat":
        return "resting_period"
    else:  # tidak cocok tanam
        return "kt_not_suitable"

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
                "summary_collection": "holt-winter-summary",
                "kt_evaluations": summary_result.get("kt_evaluations", {})
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