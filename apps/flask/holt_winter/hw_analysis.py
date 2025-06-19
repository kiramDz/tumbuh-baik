import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from pymongo import MongoClient

# Koneksi ke MongoDB
client = MongoClient("mongodb://host.docker.internal:27017/")
db = client["tugas_akhir"]
collection_input = db["bmkg-api"]  # hasil prediksi
collection_output = db["hw_results"]  # collection baru untuk hasil prediksi

def run_hw_analysis():
    try:
        # Test koneksi dulu
        print("Testing MongoDB connection...")
        count = collection_input.count_documents({})
        print(f"Found {count} documents in bmkg-api collection")
        
        # Fetch sample data untuk test
        sample_data = list(collection_input.find({}).limit(5))
        print(f"Sample data: {sample_data}")
        
        return {"status": "success", "document_count": count}
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise e
