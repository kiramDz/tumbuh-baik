from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import itertools
import warnings
warnings.filterwarnings('ignore')

def create_temporal_features(dates):
    """
    Membuat fitur temporal untuk model klasifikasi
    """
    df = pd.DataFrame({'date': dates})
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_dry_season'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(int)  # Jun-Sep
    df['is_wet_season'] = ((df['month'] >= 12) | (df['month'] <= 3)).astype(int)  # Dec-Mar
    
    # Cyclical encoding untuk musiman
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Lag features (hujan kemarin)
    return df.drop('date', axis=1)

def train_rain_occurrence_model(rain_data, dates):
    """
    Bagian 1: Model klasifikasi untuk kejadian hujan (0/1)
    """
    print("\n=== Training Rain Occurrence Model ===")
    
    # Binary target: 1 jika hujan (>0.1mm), 0 jika tidak
    rain_binary = (rain_data > 0.1).astype(int)
    
    # Create features
    features = create_temporal_features(dates)
    
    # Add lag features
    features['rain_yesterday'] = rain_binary.shift(1).fillna(0)
    features['rain_2days_ago'] = rain_binary.shift(2).fillna(0)
    features['rain_3days_ago'] = rain_binary.shift(3).fillna(0)
    features['rain_last_week'] = rain_binary.rolling(7).mean().shift(1).fillna(0)
    
    # Remove first few rows due to lag features
    valid_idx = 3
    X = features.iloc[valid_idx:]
    y = rain_binary.iloc[valid_idx:]
    
    print(f"Training data: {len(X)} samples")
    print(f"Rain days: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"No-rain days: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Split data
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Grid search untuk Random Forest Classifier
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_classifier = grid_search.best_estimator_
    
    # Evaluate classifier
    y_pred = best_classifier.predict(X_test)
    y_pred_proba = best_classifier.predict_proba(X_test)[:, 1]
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))
    
    return best_classifier, X.columns.tolist()

def train_rain_intensity_model(rain_data, dates):
    """
    Bagian 2: Model regresi untuk intensitas hujan (hanya hari hujan)
    """
    print("\n=== Training Rain Intensity Model ===")
    
    # Filter hanya hari hujan (>0.1mm)
    rain_mask = rain_data > 0.1
    rain_intensity = rain_data[rain_mask]
    rain_dates = dates[rain_mask]
    
    print(f"Rainy days for intensity model: {len(rain_intensity)}")
    print(f"Intensity stats: min={rain_intensity.min():.2f}, max={rain_intensity.max():.2f}, mean={rain_intensity.mean():.2f}")
    
    if len(rain_intensity) < 50:
        print("⚠️ Insufficient rainy days, using simple average")
        return None, rain_intensity.mean()
    
    # Try Holt-Winters untuk intensitas hujan
    try:
        # Grid search untuk intensity model
        best_score = float('inf')
        best_model = None
        best_params = None
        
        alpha_range = [0.1, 0.3, 0.5]
        beta_range = [0.1, 0.3]
        gamma_range = [0.1, 0.3]
        seasonal_periods_options = [30, 90] if len(rain_intensity) >= 180 else [30] if len(rain_intensity) >= 60 else []
        
        # Split data
        split_point = int(len(rain_intensity) * 0.8)
        train_intensity = rain_intensity.iloc[:split_point]
        test_intensity = rain_intensity.iloc[split_point:]
        
        if len(seasonal_periods_options) > 0:
            for seasonal_periods in seasonal_periods_options:
                if len(train_intensity) < seasonal_periods * 2:
                    continue
                    
                for alpha in alpha_range:
                    for beta in beta_range:
                        for gamma in gamma_range:
                            try:
                                model = ExponentialSmoothing(
                                    train_intensity,
                                    trend="add",
                                    seasonal="add",
                                    seasonal_periods=seasonal_periods
                                ).fit(
                                    smoothing_level=alpha,
                                    smoothing_trend=beta,
                                    smoothing_seasonal=gamma,
                                    optimized=False
                                )
                                
                                forecast = model.forecast(len(test_intensity))
                                mae = mean_absolute_error(test_intensity, forecast)
                                
                                if mae < best_score:
                                    best_score = mae
                                    best_model = model
                                    best_params = {
                                        'alpha': alpha,
                                        'beta': beta,
                                        'gamma': gamma,
                                        'seasonal_periods': seasonal_periods
                                    }
                            except:
                                continue
        
        # Fallback to simple exponential smoothing
        if best_model is None:
            print("Using simple exponential smoothing for intensity")
            best_model = ExponentialSmoothing(
                rain_intensity,
                trend=None,
                seasonal=None
            ).fit(smoothing_level=0.3)
            best_params = {'alpha': 0.3, 'model_type': 'simple'}
        
        print(f"Best intensity model parameters: {best_params}")
        print(f"Intensity model MAE: {best_score:.3f}")
        
        return best_model, best_params
        
    except Exception as e:
        print(f"Intensity model failed: {e}")
        return None, rain_intensity.mean()

def two_part_forecast(occurrence_model, intensity_model, feature_columns, 
                     last_rain_data, forecast_dates, intensity_params):
    """
    Generate forecast menggunakan two-part model
    """
    print(f"\n=== Generating Two-Part Forecast for {len(forecast_dates)} days ===")
    
    forecasts = []
    rain_history = last_rain_data.copy()
    
    for i, forecast_date in enumerate(forecast_dates):
        # Create features untuk hari ini
        feature_row = create_temporal_features(pd.Series([forecast_date]))
        
        # Add lag features
        feature_row['rain_yesterday'] = 1 if len(rain_history) > 0 and rain_history.iloc[-1] > 0.1 else 0
        feature_row['rain_2days_ago'] = 1 if len(rain_history) > 1 and rain_history.iloc[-2] > 0.1 else 0
        feature_row['rain_3days_ago'] = 1 if len(rain_history) > 2 and rain_history.iloc[-3] > 0.1 else 0
        
        # Rolling average
        if len(rain_history) >= 7:
            feature_row['rain_last_week'] = (rain_history.iloc[-7:] > 0.1).mean()
        else:
            feature_row['rain_last_week'] = (rain_history > 0.1).mean() if len(rain_history) > 0 else 0
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in feature_row.columns:
                feature_row[col] = 0
        
        # Reorder columns to match training
        feature_row = feature_row[feature_columns]
        
        # Part 1: Predict rain occurrence
        rain_prob = occurrence_model.predict_proba(feature_row)[0, 1]
        will_rain = rain_prob > 0.5  # Threshold bisa disesuaikan
        
        # Part 2: Predict rain intensity if it will rain
        if will_rain and intensity_model is not None:
            if hasattr(intensity_model, 'forecast'):
                # Holt-Winters model
                try:
                    intensity = intensity_model.forecast(1)[0]
                    intensity = max(0.1, intensity)  # Minimal 0.1mm jika hujan
                except:
                    intensity = intensity_params if isinstance(intensity_params, (int, float)) else 5.0
            else:
                # Fallback to average
                intensity = intensity_params if isinstance(intensity_params, (int, float)) else 5.0
        else:
            intensity = 0.0
        
        # Cap maximum intensity
        intensity = min(intensity, 200)  # Max 200mm/day
        
        forecasts.append({
            'date': forecast_date,
            'rain_probability': rain_prob,
            'will_rain': will_rain,
            'intensity': intensity,
            'final_forecast': intensity
        })
        
        # Update rain history for next iteration
        rain_history = pd.concat([rain_history, pd.Series([intensity])]).iloc[-30:]  # Keep last 30 days
    
    forecast_df = pd.DataFrame(forecasts)
    
    print(f"Forecast summary:")
    print(f"  Rainy days predicted: {forecast_df['will_rain'].sum()} ({forecast_df['will_rain'].mean()*100:.1f}%)")
    print(f"  Average rain probability: {forecast_df['rain_probability'].mean():.3f}")
    print(f"  Average intensity (when raining): {forecast_df[forecast_df['will_rain']]['intensity'].mean():.2f}mm")
    print(f"  Total predicted rainfall: {forecast_df['final_forecast'].sum():.2f}mm")
    
    return forecast_df

def run_optimized_hw_analysis(collection_name, target_column, save_collection="holt-winter", config_id=None, append_column_id=True, client=None):
    """
    Fungsi Two-Part Model untuk prediksi curah hujan
    """
    print(f"=== Start Two-Part Rain Model Analysis for {collection_name}.{target_column} ===")
    
    # MongoDB connection
    if client is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        MONGO_URI = os.getenv("MONGODB_URI")
        client = MongoClient(MONGO_URI)
        should_close_client = True
    else:
        should_close_client = False
    
    db = client["tugas_akhir"]
    
    try:
        # Fetch data dari collection yang ditentukan
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"Fetched {len(source_data)} records from {collection_name}")
        
        if not source_data:
            raise ValueError(f"No data found in collection {collection_name}")
        
        # Prepare DataFrame
        df = pd.DataFrame(source_data)
        
        # Cek apakah ada kolom Date/timestamp
        date_column = None
        for col in ['Date', 'date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            raise ValueError(f"No date column found in {collection_name}")
        
        # Set timestamp index
        df['timestamp'] = pd.to_datetime(df[date_column])
        df.set_index('timestamp', inplace=True)

        # Pastikan indeks harian tanpa duplikasi
        df = df[~df.index.duplicated(keep='first')]
        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
        df = df.reindex(date_range, fill_value=0)
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get data
        rain_data = df[target_column].fillna(0)
        dates = df.index
        
        if len(rain_data) < 365:
            raise ValueError(f"Insufficient data for {target_column}: need at least 1 year")
        
        print(f"Rain data stats:")
        print(f"  Total days: {len(rain_data)}")
        print(f"  Rainy days (>0.1mm): {(rain_data > 0.1).sum()} ({(rain_data > 0.1).mean()*100:.1f}%)")
        print(f"  Zero days: {(rain_data == 0).sum()} ({(rain_data == 0).mean()*100:.1f}%)")
        print(f"  Max rainfall: {rain_data.max():.2f}mm")
        print(f"  Average rainfall (all days): {rain_data.mean():.2f}mm")
        print(f"  Average rainfall (rainy days only): {rain_data[rain_data > 0.1].mean():.2f}mm")
        
        # Train two-part model
        occurrence_model, feature_columns = train_rain_occurrence_model(rain_data, dates)
        intensity_model, intensity_params = train_rain_intensity_model(rain_data, dates)
        
        # Calculate forecast horizon
        forecast_start_date = pd.Timestamp("2025-09-20")
        forecast_end_date = pd.Timestamp("2026-09-19")
        forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
        
        print(f"Forecast horizon: {len(forecast_dates)} days")
        
        # Generate forecast
        last_30_days = rain_data.iloc[-30:]  # Last 30 days untuk context
        
        forecast_df = two_part_forecast(
            occurrence_model, intensity_model, feature_columns,
            last_30_days, forecast_dates, intensity_params
        )
        
        # Prepare forecast documents
        forecast_docs = []
        
        for _, row in forecast_df.iterrows():
            forecast_date_only = datetime.strptime(row['date'].strftime('%Y-%m-%d'), '%Y-%m-%d')
            
            doc = {
                "forecast_date": forecast_date_only,
                "timestamp": datetime.now().isoformat(),
                "source_collection": collection_name,
                "config_id": config_id,
                "parameters": {
                    target_column: {
                        "forecast_value": float(row['final_forecast']),
                        "rain_probability": float(row['rain_probability']),
                        "will_rain": bool(row['will_rain']),
                        "model_metadata": {
                            "model_type": "two_part",
                            "occurrence_model": "RandomForest",
                            "intensity_model": "HoltWinters" if intensity_model is not None else "Average",
                            "intensity_params": intensity_params
                        }
                    }
                }
            }

            if append_column_id:
                doc["column_id"] = f"{collection_name}_{target_column}"

            forecast_docs.append(doc)
        
        # Upsert ke collection
        upsert_count = 0
        for doc in forecast_docs:
            result = db[save_collection].update_one(
                {"forecast_date": doc["forecast_date"]},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1
        
        print(f"✓ Upserted {upsert_count} forecast documents to {save_collection}")
        
        result_summary = {
            "collection_name": collection_name,
            "target_column": target_column,
            "forecast_days": len(forecast_docs),
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_type": "two_part",
            "model_summary": {
                "total_rainy_days_predicted": int(forecast_df['will_rain'].sum()),
                "total_rainfall_predicted": float(forecast_df['final_forecast'].sum()),
                "average_rain_probability": float(forecast_df['rain_probability'].mean()),
                "rainy_days_percentage": float(forecast_df['will_rain'].mean() * 100)
            },
            "forecast_range": {
                "min": float(forecast_df['final_forecast'].min()),
                "max": float(forecast_df['final_forecast'].max())
            }
        }
        
        print(f"✓ Two-Part Model analysis completed for {collection_name}.{target_column}")
        return result_summary
        
    except Exception as e:
        print(f"❌ Error in Two-Part Model analysis: {str(e)}")
        raise e
    
    finally:
        if should_close_client:
            client.close()

if __name__ == "__main__":
    run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )