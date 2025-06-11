from pymongo import MongoClient
import pandas as pd
from datetime import datetime

def generate_tanam_summary():
    client = MongoClient("mongodb://host.docker.internal:27017/")
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
        if row["curah_hujan"] > 150:  # Threshold bisa kamu sesuaikan
            return "tidak cocok tanam"
        elif row["curah_hujan"] < 50 and row["kelembapan"] > 70:
            return "sangat cocok tanam"
        else:
            return "cocok tanam"

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

    db["bmkg-tanam-summary"].delete_many({})  # Optional: bersihkan dulu
    db["bmkg-tanam-summary"].insert_many(docs_to_insert)

    return {"message": "Tanam summary saved", "total_months": len(docs_to_insert)}
