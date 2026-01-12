from datetime import datetime, timedelta
from pymongo import MongoClient
import calendar

KT_WINDOWS = {
    "KT-1": {"start": (9, 20), "end": (10, 10), "year_offset": -1},     # 20 Sep – 10 Okt (tahun sebelumnya)
    "KT-2": {"start": (2, 20), "end": (3, 10), "year_offset": 0},       # 20 Feb – 10 Mar (tahun sama)
    "KT-3": {"start": (7, 20), "end": (8, 10), "year_offset": 0},       # 20 Jul – 10 Ags (tahun sama)
}

def generate_monthly_summary(config_id, client=None):
    if client is None:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        import os
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_URI"))
    
    db = client.get_default_database()
    forecast_data = list(db["holt-winter"].find({"config_id": config_id}))

    if not forecast_data:
        return {"success": False, "error": "No forecast data found."}

    grouped = group_forecast_by_date(forecast_data)
    forecast_years = sorted(set(d.year for d in grouped.keys()))
    
    summaries = []
    kt_results = {year: [] for year in forecast_years}

    for year in forecast_years:
        for kt_label, window in KT_WINDOWS.items():
            kt_year = year + window["year_offset"]
            start_range = get_window_dates(kt_year, window["start"][0], window["start"][1])
            end_range = get_window_dates(kt_year, window["end"][0], window["end"][1])

            if not is_kt_data_available(grouped, start_range.date(), end_range.date()):
                continue

            ideal_date = find_ideal_planting_date(grouped, start_range.date(), end_range.date())

            if ideal_date:
                # Periode KT cocok
                values_30d = [
                    grouped.get(ideal_date.date() + timedelta(days=i), {})
                    for i in range(30)
                ]
                rr_values = [v.get("RR_imputed", 0) for v in values_30d]
                rr_sum = sum(rr_values)
                rr_avg = rr_sum / len(rr_values)

                summaries.append({
                    "month": ideal_date.strftime("%Y-%m"),
                    "start_date": ideal_date,
                    "end_date": ideal_date + timedelta(days=160),
                    "kt_period": f"{kt_label}-{kt_year}",
                    "status": "cocok",
                    "reason": f"Curah hujan cukup (total: {rr_sum:.2f} mm, rata-rata: {rr_avg:.2f} mm) dari {ideal_date.date()} hingga {(ideal_date + timedelta(days=29)).date()}",
                    "parameters": summarize_parameters_monthly(grouped, ideal_date),
                    "config_id": config_id,     
                })
                kt_results[year].append("cocok")
            else:
                # Coba cari periode non-KT
                non_kt_date = find_non_kt_planting_date(grouped, end_range.date())
                if non_kt_date:
                    values_30d = [
                        grouped.get(non_kt_date.date() + timedelta(days=i), {})
                        for i in range(30)
                    ]
                    rr_values = [v.get("RR_imputed", 0) for v in values_30d]
                    rr_sum = sum(rr_values)
                    rr_avg = rr_sum / len(rr_values)

                    summaries.append({
                        "month": non_kt_date.strftime("%Y-%m"),
                        "start_date": non_kt_date,
                        "end_date": non_kt_date + timedelta(days=160),
                        "kt_period": f"{kt_label}-{kt_year}-non_kt",
                        "status": "cocok_non_kt",
                        "reason": f"Curah hujan cukup di luar KT (total: {rr_sum:.2f} mm, rata-rata: {rr_avg:.2f} mm) dari {non_kt_date.date()} hingga {(non_kt_date + timedelta(days=29)).date()}",
                        "parameters": summarize_parameters_monthly(grouped, non_kt_date),
                        "config_id": config_id,
                    })
                    kt_results[year].append("cocok_non_kt")
                else:
                    summaries.append({
                        "month": end_range.strftime("%Y-%m"),
                        "start_date": end_range,  # Default 10 Okt untuk KT-1
                        "end_date": end_range + timedelta(days=160),
                        "kt_period": f"{kt_label}-{kt_year}",
                        "status": "rehat",
                        "reason": f"Curah hujan tidak mencukupi 130-300mm dalam 30 hari berturut-turut dari {start_range.date()} sampai {end_range.date()} atau periode cadangan.",
                        "parameters": summarize_parameters_monthly(grouped, end_range),
                        "config_id": config_id,
                    })
                    kt_results[year].append("rehat")
    # Cek jika semua KT tidak cocok, cari periode fallback tahunan
    for year in forecast_years:
        if all(status == "rehat" for status in kt_results[year]):
            kt_ranges = [
                (get_window_dates(year + KT_WINDOWS[kt]["year_offset"], KT_WINDOWS[kt]["start"][0], KT_WINDOWS[kt]["start"][1]).date(),
                 get_window_dates(year + KT_WINDOWS[kt]["year_offset"], KT_WINDOWS[kt]["end"][0], KT_WINDOWS[kt]["end"][1]).date())
                for kt in KT_WINDOWS
            ]
            fallback_date = find_yearly_fallback_planting_date(grouped, year, kt_ranges)
            if fallback_date:
                values_30d = [
                    grouped.get(fallback_date.date() + timedelta(days=i), {})
                    for i in range(30)
                ]
                rr_values = [v.get("RR_imputed", 0) for v in values_30d]
                rr_sum = sum(rr_values)
                rr_avg = rr_sum / len(rr_values)

                summaries.append({
                    "month": fallback_date.strftime("%Y-%m"),
                    "start_date": fallback_date,
                    "end_date": fallback_date + timedelta(days=160),
                    "kt_period": f"non_kt-{year}",
                    "status": "cocok_non_kt",
                    "reason": f"Curah hujan cukup di luar semua KT (total: {rr_sum:.2f} mm, rata-rata: {rr_avg:.2f} mm) dari {fallback_date.date()} hingga {(fallback_date + timedelta(days=29)).date()}",
                    "parameters": summarize_parameters_monthly(grouped, fallback_date),
                    "config_id": config_id,
                })

    if summaries:
        db["holt-winter-summary"].delete_many({"config_id": config_id})
        db["holt-winter-summary"].insert_many(summaries)
        return {"success": True, "summaries_generated": len(summaries)}
    
    return {"success": False, "error": "No summaries generated."}

def is_kt_data_available(grouped, start_date, end_date):
    """
    Cek apakah data untuk rentang KT tersedia
    Minimal 80% dari rentang tanggal harus ada datanya
    """
    total_days = (end_date - start_date).days + 1
    available_days = sum(1 for d in grouped.keys() if start_date <= d <= end_date)
    
    return available_days >= (total_days * 0.8)


def get_window_dates(year, month, day):
    return datetime(year, month, day)

def group_forecast_by_date(forecast_data):
    grouped = {}
    for doc in forecast_data:
        date = doc["forecast_date"].date()
        for param, content in doc["parameters"].items():
            val = content.get("forecast_value", 0)
            if date not in grouped:
                grouped[date] = {}
            grouped[date][param] = val
    return grouped

def find_ideal_planting_date(grouped, start_date, end_date):
    """
    Cari tanggal mulai tanam dengan akumulasi RR 130-150mm dalam 30 hari berturut-turut
    yang dimulai dari rentang KT (start_date sampai end_date)
    
    Contoh KT-1: 20 Sep - 10 Okt
    - Test: 20 Sep - 20 Okt (30 hari)
    - Test: 21 Sep - 21 Okt (30 hari)  
    - Test: 22 Sep - 22 Okt (30 hari)
    - ...
    - Test: 10 Okt - 10 Nov (30 hari) <- batas maksimal
    """
    # Iterasi setiap tanggal dalam rentang KT sebagai kemungkinan tanggal mulai tanam
    current_date = start_date
    
    while current_date <= end_date:
        # Cek 30 hari berturut-turut mulai dari current_date
        total_rr = 0
        valid_days = 0
        
        for i in range(30):
            check_date = current_date + timedelta(days=i)
            rr_value = grouped.get(check_date, {}).get("RR_imputed", 0)
            if rr_value is not None:
                total_rr += rr_value
                valid_days += 1
        
        # PERBAIKAN: Cari total curah hujan 130-150mm dalam 30 hari (bukan rata-rata)
        if total_rr >= 130 and total_rr <= 300 and valid_days >= 25:  # Minimal 25 hari ada data, max 300mm untuk hindari banjir
            return datetime.combine(current_date, datetime.min.time())
        
        # Lanjut ke tanggal berikutnya
        current_date += timedelta(days=1)
    
    return None

def summarize_parameters_monthly(grouped, start_date):
    """
    Buat summary parameter per bulan selama 4 bulan (160 hari)
    Dinamis untuk semua parameter yang ada
    """
    if start_date is None:
        return {}
    
    # Convert datetime ke date jika perlu
    if hasattr(start_date, 'date'):
        start_date = start_date.date()
    
    result = {}
    
    # Ambil semua parameter yang tersedia (dinamis)
    all_params = set()
    for date_data in grouped.values():
        all_params.update(date_data.keys())
    
    # Hitung per bulan selama 4 bulan
    for month_offset in range(4):
        month_start_date = start_date + timedelta(days=month_offset * 30)
        month_label = f"month_{month_offset + 1}"
        
        month_data = {}
        for param in all_params:
            total = 0
            count = 0
            for i in range(30):
                date = month_start_date + timedelta(days=i)
                val = grouped.get(date, {}).get(param)
                if val is not None:
                    total += val
                    count += 1
            
            if count > 0:
                month_data[param] = round(total / count, 2)
        
        result[month_label] = month_data
    
    return result


# FALLBACK 
def find_non_kt_planting_date(grouped, original_end_date, extra_days=60):
    """
    Cari periode 30 hari dengan RR >= 130 mm di luar rentang KT formal.
    Misal, untuk KT-1: cek dari 11 Okt hingga 10 Des (60 hari tambahan).
    """
    start_date = original_end_date + timedelta(days=1)
    end_date = start_date + timedelta(days=extra_days)
    
    current_date = start_date
    while current_date <= end_date:
        total_rr = 0
        valid_days = 0
        for i in range(30):
            check_date = current_date + timedelta(days=i)
            rr_value = grouped.get(check_date, {}).get("RR_imputed", 0)
            if rr_value is not None:
                total_rr += rr_value
                valid_days += 1
        if total_rr >= 130 and total_rr <= 300 and valid_days >= 25:
            return datetime.combine(current_date, datetime.min.time())
        current_date += timedelta(days=1)
    return None

def find_yearly_fallback_planting_date(grouped, year, exclude_kt_ranges):
    """
    Cari periode 30 hari dengan RR >= 130 mm dalam setahun, kecuali rentang KT formal.
    exclude_kt_ranges: List rentang KT formal untuk dihindari [(start_date, end_date), ...]
    """
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year, 12, 31)
    current_date = start_of_year

    while current_date <= end_of_year - timedelta(days=30):
        # Konversi current_date ke datetime.date untuk perbandingan
        current_date_date = current_date.date()
        in_kt_range = False
        for kt_start, kt_end in exclude_kt_ranges:
            if kt_start <= current_date_date <= kt_end:
                in_kt_range = True
                break
        if in_kt_range:
            current_date += timedelta(days=1)
            continue

        total_rr = 0
        valid_days = 0
        for i in range(30):
            check_date = current_date + timedelta(days=i)
            rr_value = grouped.get(check_date.date(), {}).get("RR_imputed", 0)
            if rr_value is not None:
                total_rr += rr_value
                valid_days += 1
        if total_rr >= 130 and total_rr <= 300 and valid_days >= 25:
            return datetime.combine(current_date, datetime.min.time())
        current_date += timedelta(days=1)
    return None