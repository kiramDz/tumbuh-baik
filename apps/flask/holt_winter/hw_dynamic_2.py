
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
    forecast = np.array(forecast)
    if param_name in ["RR", "RR_imputed"]: 
        if lam is not None:  # Balikkan Box-Cox
            forecast = (forecast * lam + 1) ** (1/lam) - 1
        forecast = np.clip(forecast, 0, 300)
    elif param_name == "NDVI":  
        forecast = np.clip(forecast, -1, 1)  # Rentang NDVI
    elif param_name in ["RH_AVG", "RH_AVG_preprocessed"]:  
        forecast = np.clip(forecast, 0, 100)  
    elif param_name in ["TAVG", "TMAX", "TMIN"]:  
        forecast = np.clip(forecast, 10, 50)  # Celsius, rentang realistis
    elif param_name in ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]:
        # Radiasi Matahari (W/m²), tidak mungkin negatif
        forecast = np.clip(forecast, 0, 1400) 
    else:
        print(f"Warning: No post-processing defined for {param_name}")
    return forecast


def detect_seasonal_period(data, param_name):
    """
    Deteksi periode musiman menggunakan seasonal_decompose
    """
    is_ndvi = param_name in ["NDVI"]
    
    if is_ndvi:
        min_period = 4
        max_period = len(data) // 2
        periods = range(min_period, min(max_period, 23))
        best_period = min_period
        best_residual = float('inf')

        for period in periods:
            if period >= len(data):
                continue
            try:
                result = seasonal_decompose(data, model='additive', period=period, extrapolate_trend='freq')
                residual = np.nanmean(np.abs(result.resid))
                if residual < best_residual:
                    best_residual = residual
                    best_period = period
            except Exception:
                continue
        return best_period
    else:
        return 365  
    


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
        if len(train_split) < seasonal_periods * 2:
            continue
            
        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    try:
                        print(f"🔧 Trying: alpha={alpha}, beta={beta}, gamma={gamma}, season={seasonal_periods}")
                        # Fit model
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
                        
                        # Forecast
                        forecast = model.forecast(len(val_split))
                        forecast = post_process_forecast(forecast, param_name)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(val_split, forecast)
                        mad = np.mean(np.abs(val_split - np.mean(val_split)))
                        mse = mean_squared_error(val_split, forecast)
                        mape = np.mean(np.abs((val_split - forecast) / np.where(val_split != 0, val_split, 1))) * 100
                        rmse = np.sqrt(mse)
                        score = mae * 0.7 + rmse * 0.3
                        
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'seasonal_periods': seasonal_periods
                            }
                            best_metrics = {
                                'mae': mae,
                                'mad': mad,
                                'mape': mape,
                                'mse': mse,
                                'rmse': rmse,
                                'valid_models': valid_models + 1
                            }
                            print(f"✅ New best found! Score: {score:.4f}, Params: {best_params}")
                            valid_models += 1
                            
                    except Exception as e:
                        continue
    if best_params:
        print(f"\n🎯 Best Params: {best_params}")
        print(f"📈 Metrics: {best_metrics}")
    else:
        print("❌ No valid model found.")

    return best_params, best_metrics



def run_optimized_hw_analysis(collection_name, target_column, save_collection="holt-winter", config_id=None, append_column_id=True, client=None, start_date=None, end_date=None):
    """
    Fungsi Holt-Winter yang dinamis berdasarkan parameter dari forecast_config
    """
    print(f"=== Start Holt-Winter Analysis for {collection_name}.{target_column} ===")
    
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

        # Tentukan frekuensi berdasarkan parameter
        is_ndvi = target_column in ["NDVI", "NDVI_imputed"]
        freq = '16D' if is_ndvi else 'D'

        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        missing_dates = date_range.difference(df.index)
        print(f"Missing dates: {missing_dates}")

         # Reindex dengan interpolasi untuk NDVI, fill_value=0 untuk lainnya
        if is_ndvi:
            df = df.reindex(date_range).interpolate(method='linear')
        else:
            df = df.reindex(date_range).interpolate(method='linear')
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get data (tanpa preprocessing karena data sudah bersih)
        param_data = df[target_column].dropna()
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data for {target_column}")
        
        # Debug data
        print(f"Data summary for {target_column}:")
        print(f"Total values: {len(param_data)}")
        print(f"Zero values: {(param_data == 0).sum()}")
        print(f"Non-zero values: {(param_data > 0).sum()}")
        print(f"Mean: {param_data.mean():.3f}, Std: {param_data.std():.3f}")

        try:
            best_period = detect_seasonal_period(param_data, target_column)
            print(f"🔍 Starting decompose for {target_column}, data length: {len(param_data)}, period: {best_period}")
            
            decompose_result = seasonal_decompose(param_data, model='additive', period=best_period, extrapolate_trend='freq')
            decompose_docs = []
            for i, date in enumerate(param_data.index):
                doc = {
                    "date": date.to_pydatetime(),
                    "config_id": config_id,
                    "parameters": {
                        target_column: {
                            "trend": float(decompose_result.trend[i]) if not np.isnan(decompose_result.trend[i]) else None,
                            "seasonal": float(decompose_result.seasonal[i]) if not np.isnan(decompose_result.seasonal[i]) else None,
                            "resid": float(decompose_result.resid[i]) if not np.isnan(decompose_result.resid[i]) else None
                        }
                    }
                }
                decompose_docs.append(doc)
            
            db["temp-decompose"].insert_many(decompose_docs)
            print(f"✅ Decompose success: {len(decompose_docs)} documents saved to temp-decompose")
            
        except Exception as e:
            print(f"❌ Decompose failed for {target_column}: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        # Grid search
        best_params, error_metrics = grid_search_hw_params(param_data, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid model found for {target_column}")
        
        # Fit final model
        final_model = fit_robust_model(param_data, best_params)
        
        if final_model is None:
            raise ValueError(f"Failed to fit final model for {target_column}")
        
        if start_date is not None and end_date is not None:
            # Convert ke pandas datetime jika belum
            forecast_start_date = pd.to_datetime(start_date)
            forecast_end_date = pd.to_datetime(end_date)
            print(f"[INFO] Using custom date range: {forecast_start_date.date()} to {forecast_end_date.date()}")
        else:
            # Fallback ke logika lama (tanggal terakhir data + 1 tahun)
            last_data_date = df.index[-1]
            forecast_start_date = last_data_date + pd.Timedelta(days=1)
            forecast_end_date = forecast_start_date + pd.Timedelta(days=364)
            print(f"[INFO] Using default date range: {forecast_start_date.date()} to {forecast_end_date.date()}")
        
        date_increment = '16D' if is_ndvi else 'D'

        forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq=date_increment)
        forecast_steps = len(forecast_dates)

        
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
            target_column = list(doc["parameters"].keys())[0]  # ambil nama parameter
            result = db[save_collection].update_one(
                {
                    "forecast_date": doc["forecast_date"],
                    "config_id": doc["config_id"]
                },
                {
                    "$set": {
                        f"parameters.{target_column}": doc["parameters"][target_column],  # hanya overwrite kolom ini
                        "timestamp": doc["timestamp"],
                        "source_collection": doc["source_collection"],
                        "column_id": doc.get("column_id")
                    }
                },
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
            "error_metrics": error_metrics,  
            "forecast_range": {
                "start": forecast_start_date.strftime("%Y-%m-%d"),
                "end": forecast_end_date.strftime("%Y-%m-%d"),
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