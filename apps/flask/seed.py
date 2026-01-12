import json
from pymongo import MongoClient

client = MongoClient("mongodb://mongodb:27017")
db = client["tugas_akhir"]

def load_json_and_insert(collection_name, file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    # Coba tanpa _id untuk menghindari konflik
    for item in data:
        if "_id" in item:
            del item["_id"]
    
    try:
        result = db[collection_name].insert_many(data)
        print(f"Berhasil menyisipkan {len(result.inserted_ids)} dokumen")
    except Exception as e:
        print(f"Error: {e}")

load_json_and_insert("bmkg-api", "data/tugas_akhir_bmkg-api.json")