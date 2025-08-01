from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

def fit_robust_model(data, best_params, param_name="NDVI"):
    """
    Fit model dengan error handling yang lebih baik
    """
    print(f"🧪 Received data length: {len(data)}")
    print(f"📦 Params received: {best_params}")
    try:
        default_period = 23 if param_name == "NDVI" else 365
        model = ExponentialSmoothing(
            data,
            trend="add",
            seasonal="add",
            seasonal_periods=best_params.get("seasonal_periods", default_period),
            damped_trend=True
        ).fit(
            optimized=True
        )
        forecast_raw = model.forecast(steps=30)
        print("📈 Raw forecast (30 hari):")
        print(forecast_raw.round(2).to_list())
        return model
    except Exception as e:
        print(f"Model fitting failed: {e}")
        try:
            model = ExponentialSmoothing(
                data,
                trend=None,
                seasonal=None
            ).fit(smoothing_level=0.3)
            return model
        except:
            return None

def post_process_forecast(forecast, param_name, lam=None):
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
    elif param_name == "RH_AVG":  
        forecast = np.clip(forecast, 0, 100)  
    elif param_name in ["T_AVG", "T_MAX", "T_MIN"]:  
        forecast = np.clip(forecast, -10, 50)  # Celsius, rentang realistis
    else:
        print(f"Warning: No post-processing defined for {param_name}")
    return forecast


def detect_seasonal_period(data, param_name):
    """
    Deteksi periode musiman menggunakan seasonal_decompose
    """
    is_ndvi = param_name in ["NDVI", "NDVI_imputed"]
    
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
        return 180  # Paksa periode 180 hari untuk curah hujan
def grid_search_hw_params(train_data, param_name):
    """
    Grid search disesuaikan untuk pola curah hujan Indonesia
    """
    print(f"\n--- Grid Search for Indonesian Rainfall Pattern: {param_name} ---")
    
    # Tentukan frekuensi dan panjang minimum berdasarkan parameter
    is_ndvi = param_name in ["NDVI", "NDVI_imputed"]
    min_data_length = 46 if is_ndvi else 365  # 2 tahun untuk NDVI (~46 pengukuran), 1 tahun untuk lainnya
    seasonal_base = 23 if is_ndvi else 365    # 1 tahun: 23 pengukuran untuk NDVI, 365 hari untuk lainnya

    if len(train_data) < min_data_length:
        print(f"❌ Insufficient data (need at least {min_data_length} {'pengukuran' if is_ndvi else 'hari'})")
        return None, None
    
    # Parameter grid yang lebih konservatif untuk rainfall
    alpha_range = [0.3, 0.5, 0.7]  
    beta_range = [0.1, 0.3, 0.5]
    gamma_range = [0.3, 0.5, 0.7]

    # Hapus logika penentuan seasonal_periods_options yang lama
    best_period = detect_seasonal_period(train_data, param_name)
    seasonal_periods_options = [best_period]
    if is_ndvi:
        seasonal_periods_options.extend([best_period//2, best_period*2] if best_period > 4 else [4])
    else:
        seasonal_periods_options.extend([best_period//2, best_period*2] if best_period > 7 else [7])
        print(f"Testing seasonal periods: {seasonal_periods_options}")
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    valid_models = 0
    
   # atur proporsi data train dan validasi data
    if is_ndvi:
     val_size = max(4, int(len(train_data) * 0.25))
    else:
     val_size = min(365, int(len(train_data) * 0.25))
    split_point = len(train_data) - val_size
    
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    print(f"Train: {len(train_split)} days, Validation: {len(val_split)} days")
    
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
                        mse = mean_squared_error(val_split, forecast)
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((val_split - forecast) / np.where(val_split != 0, val_split, 1))) * 100
                        
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
                                'rmse': rmse,
                                'mape': mape,
                                'mse': mse,
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
            df = df.reindex(date_range, fill_value=0)
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get data (tanpa preprocessing karena data sudah bersih)
        param_data = df[target_column].dropna()

        # if target_column in ["RR", "RR_imputed"]:
        #     param_data, lam = boxcox(param_data + 1)
        #     param_data = pd.Series(param_data, index=df.index)
        # else:
        #     lam = None
        lam = None
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data for {target_column}")
        
        # Debug data
        print(f"Data summary for {target_column}:")
        print(f"Total values: {len(param_data)}")
        print(f"Zero values: {(param_data == 0).sum()}")
        print(f"Non-zero values: {(param_data > 0).sum()}")
        print(f"Mean: {param_data.mean():.3f}, Std: {param_data.std():.3f}")

        # Grid search
        best_params, error_metrics = grid_search_hw_params(param_data, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid model found for {target_column}")
        print(f"🔎 param_data length: {len(param_data)}")
        print(f"📊 Best params: {best_params}")

        # Fit final model
        final_model = fit_robust_model(param_data, best_params, target_column)
        fitted_values = final_model.fittedvalues
        print(f"Fitted values range: {fitted_values.min():.3f} to {fitted_values.max():.3f}")
        
        if final_model is None:
            raise ValueError(f"Failed to fit final model for {target_column}")
        
        # Calculate forecast horizon (sampai akhir 2026)
        # forecast_start_date = pd.Timestamp("2025-09-20")
        # forecast_end_date = pd.Timestamp("2026-09-19")
        # forecast_days = (forecast_end_date - forecast_start_date).days + 1

        forecast_start_date = pd.Timestamp("2025-09-20")
        forecast_end_date = pd.Timestamp("2026-09-19")
        forecast_days = (forecast_end_date - forecast_start_date).days + 1
        if is_ndvi:
            forecast_steps = (forecast_days // 16) + (1 if forecast_days % 16 > 0 else 0)
            forecast_steps = max(forecast_steps, 2)  # minimal 2 titik
        else:
         forecast_steps = forecast_days

        
        print(f"Forecast horizon: {forecast_steps} {'pengukuran' if is_ndvi else 'hari'}")
        
        # Generate forecast
        print(f"Generating forecast for {target_column}...")
        try:
            forecast = final_model.forecast(steps=forecast_steps)
            print(f"Raw forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            print(f"First 10 raw forecast values: {forecast[:10].round(3).to_list()}")

            if forecast is None or len(forecast) == 0:
                raise ValueError("Forecast result is empty")
            
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            
            forecast = np.array(forecast).flatten()
            
            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values")
            
            # Post-process forecast
            forecast = post_process_forecast(forecast, target_column, lam)
            
            print(f"✓ {target_column} forecast completed")
            print(f"  Processed forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            
        except Exception as e:
            raise ValueError(f"Forecast generation failed: {str(e)}")
        
        # Prepare forecast documents
        forecast_docs = []
        date_increment = pd.Timedelta(days=16) if is_ndvi else pd.Timedelta(days=1)
        
        try:
            for i in range(len(forecast)):
                forecast_date = df.index[-1] + date_increment * (i + 1)
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
                                "seasonal_periods": best_params.get("seasonal_periods", 23 if is_ndvi else best_params.get("seasonal_periods", 7)),
                                "lambda_boxcox": lam if lam is not None else None
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