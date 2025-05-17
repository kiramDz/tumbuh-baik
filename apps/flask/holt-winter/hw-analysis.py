import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from pymongo import MongoClient

# Koneksi ke MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["nama_database"]
collection_input = db["nama_koleksi_input"]    # tempat data mentah
collection_output = db["nama_koleksi_output"]  # hasil prediksi

def run_hw_analysis():
    # Fetch data dari MongoDB
    data = list(collection_input.find({}, {"_id": 0, "tanggal": 1, "kelembaban": 1}))
    df = pd.DataFrame(data)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.set_index("tanggal")
    df = df.asfreq("D")

    # Ganti nilai anomali
    df["kelembaban"] = df["kelembaban"].replace(8888, 0)

    # Holt-Winters
    model = ExponentialSmoothing(df["kelembaban"], trend="add", seasonal="add", seasonal_periods=12)
    hw_fit = model.fit()

    forecast = hw_fit.forecast(14)  # prediksi 14 hari ke depan

    # Simpan hasil ke MongoDB
    for tanggal, nilai in forecast.items():
        collection_output.update_one(
            {"tanggal": tanggal},
            {"$set": {"tanggal": tanggal, "prediksi_kelembaban": nilai}},
            upsert=True
        )
