from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

def run_hw_analysis():
    # 1. Fetch data bersih dari MongoDB
    client = MongoClient("mongodb://host.docker.internal:27017/")
    db = client["tugas_akhir"]
    bmkg_data = list(db["bmkg-data"].find().sort("timestamp", 1))  # Urutkan by timestamp
    
    # 2. Konversi ke DataFrame
    df = pd.DataFrame(bmkg_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 3. Forecast tiap parameter
    results = {}
    parameters = ["curah_hujan", "kelembapan"]  # Bisa ditambah nanti
    
    for param in parameters:
        # Model Holt-Winters (sesuaikan seasonal_periods dari metadata)
        model = ExponentialSmoothing(
            df[param],
            trend="add",
            seasonal="add",
            seasonal_periods=365  # Ambil dari metadata jika ada
        ).fit()
        
        # Prediksi 30 hari ke depan
        forecast = model.forecast(steps=30)
        
        # Simpan hasil
        results[param] = {
            "forecast_values": forecast.tolist(),
            "model_params": {
                "alpha": model.params["smoothing_level"],
                "beta": model.params["smoothing_slope"],
                "gamma": model.params["smoothing_seasonal"]
            }
        }
    
    # 4. Simpan ke hw_result
    forecast_docs = []
    for i in range(30):  # Untuk 30 hari prediksi
        doc = {
            "timestamp": datetime.now().isoformat(),  # Waktu generate prediksi
            "forecast_date": (df.index[-1] + pd.Timedelta(days=i+1)).isoformat(),  # Tanggal yg diprediksi
            "parameters": {
                param: {
                    "forecast_value": results[param]["forecast_values"][i],
                    "model_metadata": results[param]["model_params"]
                }
                for param in parameters
            }
        }
        forecast_docs.append(doc)
    
    db["hw_result"].insert_many(forecast_docs)
    return {"status": "success", "forecast_days": 30}