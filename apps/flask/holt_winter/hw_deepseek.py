from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

def run_hw_analysis():
    print("=== Start Holt-Winter Analysis ===")
    client = MongoClient("mongodb://host.docker.internal:27017/")
    db = client["tugas_akhir"]

    bmkg_data = list(db["bmkg-data"].find().sort("Date", 1))
    print(f"Fetched {len(bmkg_data)} BMKG records")

    df = pd.DataFrame(bmkg_data)
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.set_index('timestamp', inplace=True)

    results = {}
    parameters = ["RR", "RH_AVG"]  # curah hujan & kelembapan

    for param in parameters:
        print(f"Running HW model for: {param}")
        model = ExponentialSmoothing(
            df[param],
            trend="add",
            seasonal="add",
            seasonal_periods=365 #ini data asumsi, ntar ubah jadi dinamis aja
        ).fit()

        forecast = model.forecast(steps=30)
        print(f"{param} forecast done.")

        results[param] = {
            "forecast_values": forecast.tolist(),
            "model_params": {
                "alpha": model.params.get("smoothing_level"),
                "beta": model.params.get("smoothing_slope"),
                "gamma": model.params.get("smoothing_seasonal"),
            }
        }

    forecast_docs = []
    for i in range(30):
        doc = {
            "timestamp": datetime.now().isoformat(),
            "forecast_date": (df.index[-1] + pd.Timedelta(days=i+1)).isoformat(),
            "parameters": {
                param: {
                    "forecast_value": results[param]["forecast_values"][i],
                    "model_metadata": results[param]["model_params"]
                } for param in parameters
            }
        }
        forecast_docs.append(doc)

    db["bmkg-hw"].insert_many(forecast_docs)
    print("Inserted forecast docs into 'bmkg-hw' collection.")
    return forecast_docs
