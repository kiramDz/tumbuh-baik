import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

def analyze_data_profile(collection_name, target_column):
    print(f"🕵️ MENGANALISIS PROFIL DATA: {collection_name}.{target_column}")
    
    # 1. Koneksi DB
    load_dotenv()
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["tugas_akhir"]
    
    # 2. Fetch Data
    raw = list(db[collection_name].find().sort("Date", 1))
    df = pd.DataFrame(raw)
    
    # Cleaning Date
    date_col = next((c for c in ['Date', 'date', 'timestamp'] if c in df.columns), None)
    df['timestamp'] = pd.to_datetime(df[date_col])
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Interpolasi agar statistik valid
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx)
    
    # Analisis Kolom Target
    series = df[target_column]
    
    # --- HITUNG STATISTIK ---
    total_days = len(series)
    missing_val = series.isna().sum()
    zeros = (series == 0).sum()
    zeros_pct = (zeros / total_days) * 100
    
    # Statistik Data (Non-NaN)
    clean_series = series.dropna()
    max_val = clean_series.max()
    min_val = clean_series.min()
    mean_val = clean_series.mean()
    median_val = clean_series.median()
    std_dev = clean_series.std()
    skewness = clean_series.skew() # Kemiringan grafik
    
    # Cek "Kekeringan" (Consecutive Zeros)
    # Mencari tahu berapa lama rata-rata tidak hujan berturut-turut
    s = pd.Series(np.where(clean_series == 0, 1, 0))
    dry_spells = s.groupby((s != s.shift()).cumsum()).cumsum()
    max_dry_days = dry_spells.max()

    print("\n====== 📊 LAPORAN DATA UNTUK OPTIMASI LSTM ======")
    print(f"1. Total Data       : {total_days} hari ({total_days/365:.1f} tahun)")
    print(f"2. Missing Values   : {missing_val} ({missing_val/total_days*100:.1f}%)")
    print(f"3. Jumlah Nol (0)   : {zeros} ({zeros_pct:.1f}%) -> { '⚠️ SANGAT BANYAK' if zeros_pct > 40 else 'Normal'}")
    print(f"4. Max Value        : {max_val:.2f}")
    print(f"5. Rata-rata (Mean) : {mean_val:.2f}")
    print(f"6. Median           : {median_val:.2f}")
    print(f"7. Skewness         : {skewness:.2f} -> {'⚠️ MIRING EKSTREM' if abs(skewness) > 1 else 'Normal'}")
    print(f"8. Kemarau Terpanjang: {max_dry_days} hari berturut-turut tanpa hujan")
    print("===================================================")
    
    # Rekomendasi Awal Berdasarkan Data
    print("\n💡 REKOMENDASI SEMENTARA:")
    if abs(skewness) > 1:
        print("- WAJIB gunakan Log Transform (np.log1p) karena data miring.")
    if zeros_pct > 50:
        print("- Data jarang hujan (Sparse). Model mungkin bias ke 0.")
        print("- Pertimbangkan loss function 'Huber' atau 'MAE' daripada MSE.")
    if missing_val > 100:
        print("- Hati-hati interpolasi linear terlalu panjang.")
    
    client.close()

if __name__ == "__main__":
    # Ganti sesuai data Anda
    analyze_data_profile("bmkg-data", "RR")