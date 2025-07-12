from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')

def preprocess_weather_data(data, param_name):
    """
    Preprocessing khusus untuk data cuaca
    """
    print(f"\n--- Preprocessing {param_name} ---")
    print(f"Original data shape: {len(data)}")
    print(f"Zero values: {(data == 0).sum()} ({(data == 0).mean()*100:.1f}%)")
    print(f"Negative values: {(data < 0).sum()}")
    print(f"Data range: {data.min():.3f} to {data.max():.3f}")
    
    # Handle missing values
    data = data.dropna()
    
    if param_name == "RR":  # Curah Hujan
        # RR boleh 0 (tidak hujan), tapi tidak boleh negatif
        data = data.clip(lower=0)
        
        # Jika terlalu banyak 0, tambahkan smoothing kecil untuk model
        zero_ratio = (data == 0).mean()
        if zero_ratio > 0.3:  # Jika >30% adalah 0
            print(f"Warning: {zero_ratio*100:.1f}% zero values in RR data")
            # Tambahkan noise kecil untuk menghindari masalah numerical
            data = data + np.random.normal(0, 0.01, len(data))
            data = data.clip(lower=0)  # Pastikan tetap non-negatif
            
    elif param_name == "RH_AVG":  # Kelembapan
        # Kelembapan harus 0-100%
        data = data.clip(lower=0, upper=100)
        
        # Jika ada nilai tidak masuk akal, replace dengan median
        median_val = data.median()
        data = data.where((data >= 20) & (data <= 100), median_val)
    
    # Remove extreme outliers menggunakan IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Untuk RR, lower_bound minimal 0
    if param_name == "RR":
        lower_bound = max(0, lower_bound)
    
    outliers_before = len(data)
    data = data.clip(lower=lower_bound, upper=upper_bound)
    
    print(f"After preprocessing:")
    print(f"  Data range: {data.min():.3f} to {data.max():.3f}")
    print(f"  Zero values: {(data == 0).sum()} ({(data == 0).mean()*100:.1f}%)")
    print(f"  Final shape: {len(data)}")
    
    return data

def select_model_type(data, param_name):
    """
    Pilih tipe model berdasarkan karakteristik data
    """
    zero_ratio = (data == 0).mean()
    
    if param_name == "RR" and zero_ratio > 0.4:
        # Jika curah hujan banyak 0, gunakan model tanpa seasonal
        return "simple"
    elif len(data) < 730:  # Kurang dari 2 tahun data
        return "simple" 
    else:
        return "seasonal"

def fit_robust_model(data, param_name, best_params):
    """
    Fit model dengan error handling yang lebih baik
    """
    model_type = select_model_type(data, param_name)
    
    try:
        if model_type == "seasonal" and best_params.get('use_seasonal', True):
            model = ExponentialSmoothing(
                data,
                trend="add",
                seasonal="add",
                seasonal_periods=min(365, len(data)//3)  # Batasi seasonal period
            ).fit(
                smoothing_level=best_params['alpha'],
                smoothing_trend=best_params['beta'],
                smoothing_seasonal=best_params['gamma'],
                optimized=False
            )
        else:
            # Model sederhana tanpa seasonal
            model = ExponentialSmoothing(
                data,
                trend="add",
                seasonal=None
            ).fit(
                smoothing_level=best_params['alpha'],
                smoothing_trend=best_params['beta'],
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
    if param_name == "RR":  # Curah Hujan
        # Tidak boleh negatif
        forecast = np.maximum(forecast, 0)
        # Batasi maksimal (misal 200mm/hari untuk ekstrem)
        forecast = np.minimum(forecast, 200)
        
    elif param_name == "RH_AVG":  # Kelembapan
        # Harus dalam range 0-100%
        forecast = np.clip(forecast, 0, 100)
    
    elif param_name == "NDVI":  # Normalized Difference Vegetation Index
        # NDVI harus dalam range -1 to 1, tapi biasanya 0-1 untuk vegetasi
        forecast = np.clip(forecast, -1, 1)
    
    elif "Suhu" in param_name or "Temperature" in param_name:  # Suhu
        # Batasi suhu dalam range yang masuk akal (-50°C to 60°C)
        forecast = np.clip(forecast, -50, 60)
    
    return forecast

def grid_search_hw_params(train_data, param_name):
    """
    Grid search dengan preprocessing yang lebih baik
    """
    print(f"\n--- Grid Search for {param_name} ---")
    
    # Preprocessing data
    processed_data = preprocess_weather_data(train_data.copy(), param_name)
    
    if len(processed_data) < 100:
        print("❌ Insufficient data after preprocessing")
        return None, None
    
    # Parameter grid yang lebih konservatif
    alpha_range = [0.1, 0.3, 0.5, 0.7]
    beta_range = [0.1, 0.3, 0.5]
    gamma_range = [0.1, 0.3, 0.5]
    
    best_score = float('inf')
    best_params = None
    valid_models = 0
    
    # Split data for validation
    split_point = int(len(processed_data) * 0.8)
    train_split = processed_data[:split_point]
    val_split = processed_data[split_point:]
    
    print(f"Train split: {len(train_split)}, Validation split: {len(val_split)}")
    
    # Tentukan seasonal periods berdasarkan data
    if len(train_split) >= 365:
        seasonal_periods_options = [365]
    elif len(train_split) >= 30:
        seasonal_periods_options = [30]
    else:
        seasonal_periods_options = [7]
    
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
                
                # Post-process forecast
                forecast = post_process_forecast(forecast, param_name)
                
                # Skip jika masih ada masalah
                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    continue
                
                rmse = np.sqrt(mean_squared_error(val_split, forecast))
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'seasonal_periods': seasonal_periods,
                        'use_seasonal': True
                    }
                    valid_models += 1
                    print(f"✓ New best: α={alpha}, β={beta}, γ={gamma} | RMSE={rmse:.3f}")
                    
            except Exception as e:
                continue
    
    # Jika tidak ada model seasonal yang berhasil, coba simple model
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
                forecast = post_process_forecast(forecast, param_name)
                
                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    continue
                
                rmse = np.sqrt(mean_squared_error(val_split, forecast))
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': 0.1,
                        'seasonal_periods': 365,
                        'use_seasonal': False
                    }
                    valid_models += 1
                    print(f"✓ Simple model: α={alpha}, β={beta} | RMSE={rmse:.3f}")
                    
            except Exception:
                continue
    
    if best_params is None:
        print(f"❌ No valid model found for {param_name}")
        return None, None
    
    print(f"✓ Best parameters found for {param_name}: {best_params}")
    print(f"✓ Valid models tested: {valid_models}")
    
    return best_params, None

def run_optimized_hw_analysis(collection_name, target_column, save_collection="holt-winter", config_id=None, append_column_id=True, client=None):
    """
    Fungsi Holt-Winter yang dinamis berdasarkan parameter dari forecast_config
    
    Args:
        collection_name: Nama collection sumber data
        target_column: Nama kolom yang akan diforecast
        save_collection: Collection tujuan (default: "holt-winter")
        config_id: ID dari forecast_config
        append_column_id: Apakah menambahkan column ID ke metadata
        client: MongoDB client (opsional)
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
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get and preprocess data
        param_data = df[target_column].dropna()
        processed_data = preprocess_weather_data(param_data.copy(), target_column)
        
        if len(processed_data) < 100:
            raise ValueError(f"Insufficient data for {target_column} after preprocessing")
        
        # Grid search
        best_params, _ = grid_search_hw_params(processed_data, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid model found for {target_column}")
        
        # Fit final model
        final_model = fit_robust_model(processed_data, target_column, best_params)
        
        if final_model is None:
            raise ValueError(f"Failed to fit final model for {target_column}")
        
        # Calculate forecast horizon (sampai akhir 2026)
        target_end_date = pd.Timestamp('2026-12-31')
        forecast_days = (target_end_date - df.index[-1]).days
        
        # Batasi forecast horizon maksimal 2 tahun untuk stabilitas
        max_forecast_days = 730  # 2 tahun
        if forecast_days > max_forecast_days:
            forecast_days = max_forecast_days
            print(f"Forecast horizon limited to {max_forecast_days} days for stability")
        
        print(f"Forecast horizon: {forecast_days} days")
        
        # Generate forecast
        print(f"Generating forecast for {target_column}...")
        try:
            forecast = final_model.forecast(steps=forecast_days)
            
            # Validasi forecast hasil
            if forecast is None or len(forecast) == 0:
                raise ValueError("Forecast result is empty")
            
            # Convert to numpy array jika pandas Series
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            
            # Pastikan forecast adalah array 1D
            forecast = np.array(forecast).flatten()
            
            # Cek apakah ada nilai NaN atau inf
            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values")
            
            # Post-process forecast
            forecast = post_process_forecast(forecast, target_column)
            
            print(f"✓ {target_column} forecast completed")
            print(f"  Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            
        except Exception as e:
            raise ValueError(f"Forecast generation failed: {str(e)}")
        
        # Prepare forecast documents dengan struktur upsert
        forecast_docs = []
        
        try:
            for i in range(len(forecast)):  # Gunakan len(forecast) instead of forecast_days
                forecast_date = df.index[-1] + pd.Timedelta(days=i+1)
                forecast_date_str = forecast_date.strftime('%Y-%m-%d')
                
                # Validasi forecast value
                forecast_value = float(forecast[i])
                if np.isnan(forecast_value) or np.isinf(forecast_value):
                    print(f"Warning: Skipping invalid forecast value at index {i}")
                    continue
                
                # Struktur dokumen untuk upsert
                doc = {
                    "forecast_date": forecast_date_str,
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
        
        # Upsert ke collection holt-winter berdasarkan forecast_date
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
            "forecast_days": len(forecast_docs),  # Gunakan jumlah dokumen yang berhasil dibuat
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_params": best_params,
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

# Backward compatibility - fungsi lama tanpa parameter
def run_optimized_hw_analysis_old():
    """
    Fungsi lama untuk backward compatibility
    """
    return run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR",
        save_collection="bmkg-hw"
    )

if __name__ == "__main__":
    # Test dengan parameter default
    run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )