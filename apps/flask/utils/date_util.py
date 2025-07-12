def detect_date_column(db, collection_name):
    """
    Deteksi kolom tanggal dari collection dengan berbagai kemungkinan nama
    """
    try:
        sample = db[collection_name].find_one()
        if not sample:
            return None

        date_columns = [
            'Date', 'date', 'timestamp', 'Timestamp', 'created_at', 'createdAt',
            'datetime', 'DateTime', 'time', 'Time', 'tanggal', 'Tanggal',
            'waktu', 'Waktu', 'date_time', 'dateTime'
        ]

        for col in date_columns:
            if col in sample:
                print(f"✓ Found date column: {col} in {collection_name}")
                return col

        print(f"⚠️  Available columns in {collection_name}: {list(sample.keys())}")
        return None

    except Exception as e:
        print(f"❌ Error detecting date column: {e}")
        return None
