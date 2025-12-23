from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')

def fit_robust_model(data, best_params):
    """
    Fit model dengan error handling yang lebih baik
    """
    try:
        model = ExponentialSmoothing(
            data,
            trend="add",
            seasonal="add",
            seasonal_periods=best_params.get('seasonal_periods', 365)
        ).fit(
            smoothing_level=best_params['alpha'],
            smoothing_trend=best_params['beta'],
            smoothing_seasonal=best_params['gamma'],
            optimized=False
        )
        return model
    except Exception as e:
        print(f"Model fitting failed: {e}")
        # Fallback ke simple exponential smoothing
        try:
            model = ExponentialSmoothing(
                data,
                trend=None,
                seasonal=None
            ).fit(smoothing_level=0.3)
            return model
        except:
            return None

def post_process_forecast(forecast, param_name):
    """
    Post-processing untuk memastikan forecast masuk akal
    """
    if param_name == "RR" or param_name == "RR_imputed":  # Curah Hujan
        forecast = np.maximum(forecast, 0)
        forecast = np.minimum(forecast, 300)
        
    elif param_name == "RH_AVG":  # Kelembapan
        # Harus dalam range 0-100%
        forecast = np.clip(forecast, 0, 100)
    
    elif param_name == "NDVI":  # Normalized Difference Vegetation Index
        # NDVI harus dalam range -1 to 1
        forecast = np.clip(forecast, -1, 1)
    
    elif "Suhu" in param_name or "Temperature" in param_name:  # Suhu
        # Batasi suhu dalam range yang masuk akal (-50°C to 60°C)
        forecast = np.clip(forecast, -50, 60)
    
    return forecast

def grid_search_hw_params(train_data, param_name):
    """
    Grid search untuk menemukan parameter terbaik
    """
    print(f"\n--- Grid Search for {param_name} ---")
    
    if len(train_data) < 100:
        print("❌ Insufficient data for grid search")
        return None, None
    
    # Parameter grid
    alpha_range = [0.1, 0.3, 0.5, 0.7]
    beta_range = [0.1, 0.3, 0.5]
    gamma_range = [0.1, 0.3, 0.5]
    
    best_score = float('inf')
    best_params = None
    best_mae = None
    best_rmse = None
    best_mape = None
    best_mse = None
    valid_models = 0
    
    # Split data for validation
    split_point = int(len(train_data) * 0.8)
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    inferred_freq = pd.infer_freq(train_split.index)
    print(f"Inferred frequency: {inferred_freq}")
    if inferred_freq != 'D':
        print("⚠️ Non-daily frequency detected, reindexing to daily")
        date_range = pd.date_range(start=train_split.index[0], end=train_split.index[-1], freq='D')
        train_split = train_split.reindex(date_range, method='ffill')
        val_split = val_split.reindex(pd.date_range(start=val_split.index[0], end=val_split.index[-1], freq='D'), method='ffill')
    print(f"Train split: {len(train_split)}, Validation split: {len(val_split)}")
    print(f"Train min: {np.min(train_split)}, max: {np.max(train_split)}, any NaN: {np.isnan(train_split).any()}")
    print(f"Val min: {np.min(val_split)}, max: {np.max(val_split)}, any NaN: {np.isnan(val_split).any()}")
    
    # Tentukan seasonal periods berdasarkan data
    seasonal_periods_options = [365, 30, 7] if len(train_split) >= 365 else [30, 7] if len(train_split) >= 30 else [7]
    print(f"Seasonal periods to try: {seasonal_periods_options}")
    
    for seasonal_periods in seasonal_periods_options:
        for alpha, beta, gamma in itertools.product(alpha_range, beta_range, gamma_range):
            try:
                model = ExponentialSmoothing(
                    train_split,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods
                ).fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma,
                    optimized=False
                )
                
                forecast = model.forecast(steps=len(val_split))
                print(f"Forecast length: {len(forecast)}, Validation length: {len(val_split)}")

                if len(forecast) != len(val_split):
                    print(f"⚠️ Forecast length mismatch for α={alpha}, β={beta}, γ={gamma}, seasonal_periods={seasonal_periods}: expected {len(val_split)}, got {len(forecast)}")
                    continue
                
                # Post-process forecast
                forecast = post_process_forecast(forecast, param_name)

                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    print(f"⚠️ NaN or Inf in forecast for α={alpha}, β={beta}, γ={gamma}, seasonal_periods={seasonal_periods}")
                    continue
                
                # Hitung metrik evaluasi
                mse = mean_squared_error(val_split, forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(val_split, forecast)
                # Hitung MAPE, hindari pembagian dengan nol
                mape = np.mean(np.abs((val_split - forecast) / np.where(val_split != 0, val_split, 1))) * 100
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'seasonal_periods': seasonal_periods,
                        'use_seasonal': True
                    }
                    best_mae = mae
                    best_rmse = rmse
                    best_mape = mape
                    best_mse = mse
                    valid_models += 1
                    print(f"✓ New best: α={alpha}, β={beta}, γ={gamma} | RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}%, MSE={mse:.3f}")
                    
            except Exception as e:
                print(f"❌ Error for α={alpha}, β={beta}, γ={gamma}, seasonal_periods={seasonal_periods}: {str(e)}")
                continue
    # Coba model non-seasonal sebagai fallback
    if best_params is None:
        print("Trying simple model without seasonal component...")
        for alpha, beta in itertools.product(alpha_range, beta_range):
            try:
                model = ExponentialSmoothing(
                    train_split,
                    trend="add",
                    seasonal=None
                ).fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    optimized=False
                )
                
                forecast = model.forecast(steps=len(val_split))
                print(f"Simple model forecast length: {len(forecast)}, Validation length: {len(val_split)}")
                
                if len(forecast) != len(val_split):
                    print(f"⚠️ Simple model forecast length mismatch for α={alpha}, β={beta}: expected {len(val_split)}, got {len(forecast)}")
                    continue
                
                forecast = post_process_forecast(forecast, param_name)
                
                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    print(f"⚠️ NaN or Inf in simple model forecast for α={alpha}, β={beta}")
                    continue
                
                mse = mean_squared_error(val_split, forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(val_split, forecast)
                mape = np.mean(np.abs((val_split - forecast) / np.where(val_split != 0, val_split, 1))) * 100
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': 0.1,
                        'seasonal_periods': None,
                        'use_seasonal': False
                    }
                    best_mae = mae
                    best_rmse = rmse
                    best_mape = mape
                    best_mse = mse
                    valid_models += 1
                    print(f"✓ Simple model: α={alpha}, β={beta} | RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}%, MSE={mse:.3f}")
                    
            except Exception as e:
                print(f"❌ Error in simple model for α={alpha}, β={beta}: {str(e)}")
                continue

    if best_params is None:
        print("❌ No valid model found")
        return None, None
    
    print(f"✓ Best parameters found for {param_name}: {best_params}")
    print(f"✓ Valid models tested: {valid_models}")
    
    return best_params, {'mae': best_mae, 'rmse': best_rmse, 'mape': best_mape, 'mse': best_mse}

def run_optimized_hw_analysis(collection_name, target_column, save_collection="holt-winter", config_id=None, append_column_id=True, client=None):
    """
    Fungsi Holt-Winter yang dinamis berdasarkan parameter dari forecast_config
    """
    print(f"=== Start Holt-Winter Analysis for {collection_name}.{target_column} ===")
    
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
        df = df.reindex(date_range, method=0)
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get data (tanpa preprocessing karena data sudah bersih)
        param_data = df[target_column].dropna()
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data for {target_column}")
        
        # Grid search
        best_params, error_metrics = grid_search_hw_params(param_data, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid model found for {target_column}")
        
        # Fit final model
        final_model = fit_robust_model(param_data, best_params)
        
        if final_model is None:
            raise ValueError(f"Failed to fit final model for {target_column}")
        
        # Calculate forecast horizon (sampai akhir 2026)
        data_end_date = df.index[-1]
        forecast_start_date = data_end_date - pd.DateOffset(years=1)
        forecast_end_date = data_end_date + pd.DateOffset(years=1)
        forecast_days = (forecast_end_date - forecast_start_date).days + 1
        
        print(f"Forecast horizon: {forecast_days} days")
        
        # Generate forecast
        print(f"Generating forecast for {target_column}...")
        try:
            forecast = final_model.forecast(steps=forecast_days)
            
            if forecast is None or len(forecast) == 0:
                raise ValueError("Forecast result is empty")
            
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            
            forecast = np.array(forecast).flatten()
            
            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values")
            
            # Post-process forecast
            forecast = post_process_forecast(forecast, target_column)
            
            print(f"✓ {target_column} forecast completed")
            print(f"  Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            
        except Exception as e:
            raise ValueError(f"Forecast generation failed: {str(e)}")
        
        # Prepare forecast documents
        forecast_docs = []
        
        try:
            for i in range(len(forecast)):
                forecast_date = df.index[-1] + pd.Timedelta(days=i + 1)
                forecast_date_only = datetime.strptime(forecast_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                
                forecast_value = float(forecast[i])
                if np.isnan(forecast_value) or np.isinf(forecast_value):
                    print(f"Warning: Skipping invalid forecast value at index {i}")
                    continue

                doc = {
                    "forecast_date": forecast_date_only,
                    "timestamp": datetime.now().isoformat(),
                    "source_collection": collection_name,
                    "config_id": config_id,
                    "parameters": {
                        target_column: {
                            "forecast_value": forecast_value,
                            "model_metadata": {
                                "alpha": best_params["alpha"],
                                "beta": best_params["beta"],
                                "gamma": best_params["gamma"],
                                "use_seasonal": best_params.get("use_seasonal", True),
                                "seasonal_periods": best_params.get("seasonal_periods", 365)
                            }
                        }
                    }
                }

                if append_column_id:
                    doc["column_id"] = f"{collection_name}_{target_column}"

                forecast_docs.append(doc)
            
        except Exception as e:
            raise ValueError(f"Error preparing forecast documents: {str(e)}")
        
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
            "model_params": best_params,
            "error_metrics": error_metrics,  # Menambahkan MAE, RMSE, MAPE, MSE
            "forecast_range": {
                "min": float(forecast.min()),
                "max": float(forecast.max())
            }
        }
        
        print(f"✓ Analysis completed for {collection_name}.{target_column}")
        return result_summary
        
    except Exception as e:
        print(f"❌ Error in Holt-Winter analysis: {str(e)}")
        raise e
    
    finally:
        if should_close_client:
            client.close()

if __name__ == "__main__":
    run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )