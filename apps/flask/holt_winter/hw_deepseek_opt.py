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

def run_optimized_hw_analysis():
    print("=== Start Improved Holt-Winter Analysis ===")
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tugas_akhir"]
    
    # Clear existing forecasts
    deleted_count = db["bmkg-hw"].delete_many({}).deleted_count
    print(f"Cleared {deleted_count} existing forecast records")
    
    # Fetch data
    bmkg_data = list(db["bmkg-data"].find().sort("Date", 1))
    print(f"Fetched {len(bmkg_data)} BMKG records")
    
    # Prepare DataFrame
    df = pd.DataFrame(bmkg_data)
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    
    # Parameters to forecast
    parameters = ["RR", "RH_AVG"]
    
    # Calculate forecast horizon
    target_end_date = pd.Timestamp('2026-12-31')
    forecast_days = (target_end_date - df.index[-1]).days
    print(f"Forecast horizon: {forecast_days} days")
    
    results = {}
    
    for param in parameters:
        print(f"\n{'='*50}")
        print(f"Processing parameter: {param}")
        print(f"{'='*50}")
        
        # Get and preprocess data
        param_data = df[param].dropna()
        processed_data = preprocess_weather_data(param_data.copy(), param)
        
        if len(processed_data) < 100:
            print(f"❌ Insufficient data for {param}")
            continue
        
        # Grid search
        best_params, _ = grid_search_hw_params(processed_data, param)
        
        if best_params is None:
            print(f"❌ Skipping {param} - no valid model found")
            continue
        
        # Fit final model
        final_model = fit_robust_model(processed_data, param, best_params)
        
        if final_model is None:
            print(f"❌ Failed to fit final model for {param}")
            continue
        
        # Generate forecast
        print(f"Generating forecast for {param}...")
        forecast = final_model.forecast(steps=forecast_days)
        
        # Post-process forecast
        forecast = post_process_forecast(forecast, param)
        
        results[param] = {
            "forecast_values": forecast.tolist(),
            "optimal_params": best_params
        }
        
        print(f"✓ {param} forecast completed")
        print(f"  Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
    
    if not results:
        print("❌ No successful forecasts generated")
        client.close()
        return []
    
    # Prepare forecast documents
    forecast_docs = []
    
    for i in range(forecast_days):
        forecast_date = df.index[-1] + pd.Timedelta(days=i+1)
        
        doc = {
            "timestamp": datetime.now().isoformat(),
            "forecast_date": forecast_date.isoformat(),
            "parameters": {}
        }
        
        for param in results.keys():
            doc["parameters"][param] = {
                "forecast_value": results[param]["forecast_values"][i],
                "model_metadata": {
                    "alpha": results[param]["optimal_params"]["alpha"],
                    "beta": results[param]["optimal_params"]["beta"],
                    "gamma": results[param]["optimal_params"]["gamma"]
                }
            }
        
        forecast_docs.append(doc)
    
    # Insert forecast documents
    if forecast_docs:
        insert_result = db["bmkg-hw"].insert_many(forecast_docs)
        print(f"✓ Inserted {len(insert_result.inserted_ids)} forecast documents")
    
    client.close()
    print("\n✓ Analysis completed!")
    return forecast_docs

if __name__ == "__main__":
    run_optimized_hw_analysis()