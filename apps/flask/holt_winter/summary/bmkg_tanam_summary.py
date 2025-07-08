from pymongo import MongoClient
import pandas as pd
from datetime import datetime

def generate_tanam_summary():
    # client = MongoClient("mongodb://host.docker.internal:27017/")
    client = MongoClient("mongodb://localhost:27017/")  
    db = client["tugas_akhir"]
    forecast_data = list(db["bmkg-hw"].find())

    if not forecast_data:
        return {"error": "No forecast data found"}

    records = []
    for doc in forecast_data:
        date = pd.to_datetime(doc["forecast_date"])
        rr = doc["parameters"].get("RR", {}).get("forecast_value", None)
        rh = doc["parameters"].get("RH_AVG", {}).get("forecast_value", None)
        
        if rr is not None and rh is not None:
            # Validasi dan normalisasi data
            # Curah hujan tidak boleh negatif
            rr = max(0, rr)
            
            # Kelembapan harus dalam range 0-100%
            rh = max(0, min(100, rh))
            
            records.append({
                "forecast_date": date,
                "curah_hujan": rr,
                "kelembapan": rh
            })

    df = pd.DataFrame(records)
    df["month"] = df["forecast_date"].dt.to_period("M")
    
    summary = df.groupby("month").agg({
        "curah_hujan": "sum",
        "kelembapan": "mean"
    }).reset_index()

    def classify(row):
        curah_hujan = row["curah_hujan"]
        kelembapan = row["kelembapan"]
        
        # Kondisi ekstrem - tidak cocok tanam
        if curah_hujan > 400 or curah_hujan < 30:
            return "tidak cocok tanam"
        
        # Kondisi ideal untuk tanam padi
        elif 150 <= curah_hujan <= 250 and 75 <= kelembapan <= 85:
            return "sangat cocok tanam"
        
        # Kondisi baik untuk panen (curah hujan lebih rendah)
        elif 80 <= curah_hujan < 150 and 70 <= kelembapan <= 80:
            return "cocok panen"
        
        # Kondisi cukup baik untuk tanam
        elif 100 <= curah_hujan <= 300 and 70 <= kelembapan <= 90:
            return "cocok tanam"
        
        # Kondisi kurang ideal
        else:
            return "kurang cocok tanam"

    summary["status"] = summary.apply(classify, axis=1)

    # Simpan ke MongoDB
    docs_to_insert = []
    for _, row in summary.iterrows():
        docs_to_insert.append({
            "month": str(row["month"]),
            "curah_hujan_total": row["curah_hujan"],
            "kelembapan_avg": row["kelembapan"],
            "status": row["status"],
            "timestamp": datetime.now().isoformat()
        })

    for doc in docs_to_insert:
        db["bmkg-tanam-summary"].update_one(
            {"month": doc["month"]},
            {"$set": doc},
            upsert=True
        )

    return {"message": "Tanam summary saved", "total_months": len(docs_to_insert)}

# Fungsi tambahan untuk debugging
def debug_forecast_data():
    """Fungsi untuk mengecek kualitas data forecast"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tugas_akhir"]
    
    forecast_data = list(db["bmkg-hw"].find().limit(10))
    
    print("=== DEBUG FORECAST DATA ===")
    for doc in forecast_data:
        date = doc["forecast_date"]
        rr = doc["parameters"]["RR"]["forecast_value"]
        rh = doc["parameters"]["RH_AVG"]["forecast_value"]
        
        print(f"Date: {date}")
        print(f"  RR (Curah Hujan): {rr:.2f} {'❌ NEGATIF' if rr < 0 else '✓'}")
        print(f"  RH (Kelembapan): {rh:.2f} {'❌ >100%' if rh > 100 else '✓'}")
        print()
    
    client.close()
