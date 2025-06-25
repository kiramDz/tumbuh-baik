from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def time_series_split(data, test_size=0.2):
    """Split time series data temporally"""
    split_point = int(len(data) * (1 - test_size))
    return data[:split_point], data[split_point:]

def grid_search_hw_params(train_data, param_name):
    print(f"\n--- Grid Search for {param_name} ---")

    alpha_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_range = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_score = float('inf')
    best_params = None
    best_model = None

    train_split, val_split = time_series_split(train_data, test_size=0.15)

    total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range)
    current_combination = 0

    print(f"Testing {total_combinations} parameter combinations...")

    for alpha, beta, gamma in itertools.product(alpha_range, beta_range, gamma_range):
        current_combination += 1
        try:
            model = ExponentialSmoothing(
                train_split,
                trend="add",
                seasonal="add",
                seasonal_periods=365
            ).fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                smoothing_seasonal=gamma,
                optimized=False
            )

            forecast = model.forecast(steps=len(val_split))
            rmse = np.sqrt(mean_squared_error(val_split, forecast))
            mae = mean_absolute_error(val_split, forecast)
            mape = calculate_mape(val_split, forecast)

            if rmse < best_score:
                best_score = rmse
                best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
                best_model = model
                print(f"✓ New best: α={alpha}, β={beta}, γ={gamma} | RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%")
        except Exception:
            continue

        if current_combination % 25 == 0:
            print(f"Progress: {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.1f}%)")

    if best_params is None:
        print(f"❌ No valid parameter combination found for {param_name}.")
        return None, None

    print(f"\n✅ Best parameters for {param_name}:")
    print(f"α={best_params['alpha']}, β={best_params['beta']}, γ={best_params['gamma']} | Best RMSE={best_score:.3f}")
    return best_params, best_model


def calculate_forecast_horizon(start_date, end_date):
    """Calculate number of days between two dates"""
    return (end_date - start_date).days

def run_optimized_hw_analysis():
    print("=== Start Optimized Holt-Winter Analysis ===")
    
    # Connect to MongoDB via docker
    # client = MongoClient("mongodb://host.docker.internal:27017/")
    # Connect to MongoDB no docker
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tugas_akhir"]
    
    # Clear existing forecasts
    deleted_count = db["bmkg-hw"].delete_many({}).deleted_count
    print(f"Cleared {deleted_count} existing forecast records from bmkg-hw collection")
    
    # Fetch data
    bmkg_data = list(db["bmkg-data"].find().sort("Date", 1))
    print(f"Fetched {len(bmkg_data)} BMKG records")
    
    # Prepare DataFrame
    df = pd.DataFrame(bmkg_data)
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    
    # Parameters to forecast
    parameters = ["RR", "RH_AVG"]  # curah hujan & kelembapan
    
    # Calculate forecast horizon (until December 2026)
    target_end_date = pd.Timestamp('2026-12-31')
    forecast_days = calculate_forecast_horizon(df.index[-1], target_end_date)
    print(f"Forecast horizon: {forecast_days} days (until December 2026)")
    
    results = {}
    
    for param in parameters:
        print(f"\n{'='*50}")
        print(f"Processing parameter: {param}")
        print(f"{'='*50}")
        
        # Check for missing values
        param_data = df[param].dropna()
        missing_count = len(df) - len(param_data)
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values found and removed for {param}")
        
        # Grid search for optimal parameters
        best_params, _ = grid_search_hw_params(param_data, param)
        
        # Fit final model with best parameters on full dataset
        print(f"\nFitting final model for {param} with optimal parameters...")
        final_model = ExponentialSmoothing(
            param_data,
            trend="add",
            seasonal="add",
            seasonal_periods=365
        ).fit(
            smoothing_level=best_params['alpha'],
            smoothing_trend=best_params['beta'],
            smoothing_seasonal=best_params['gamma'],
            optimized=False
        )
        
        # Generate forecast
        print(f"Generating {forecast_days}-day forecast for {param}...")
        forecast = final_model.forecast(steps=forecast_days)
        
        # Calculate confidence intervals (approximate)
        residuals = final_model.resid
        forecast_std = np.std(residuals)
        confidence_interval = 1.96 * forecast_std  # 95% CI
        
        results[param] = {
            "forecast_values": forecast.tolist(),
            "optimal_params": best_params
        }
        
        print(f"✓ {param} forecast completed successfully")
    
    # Prepare forecast documents for MongoDB
    print(f"\nPreparing {forecast_days} forecast documents...")
    forecast_docs = []
    
    for i in range(forecast_days):
        forecast_date = df.index[-1] + pd.Timedelta(days=i+1)
        
        doc = {
            "timestamp": datetime.now().isoformat(),
            "forecast_date": forecast_date.isoformat(),
            "parameters": {}
        }
        
        for param in parameters:
            doc["parameters"][param] = {
                "forecast_value": results[param]["forecast_values"][i],
                "model_metadata": {
                    "alpha": results[param]["optimal_params"]["alpha"],
                    "beta": results[param]["optimal_params"]["beta"],
                    "gamma": results[param]["optimal_params"]["gamma"]
                }
            }
        
        forecast_docs.append(doc)
    if forecast_docs:
        inserted = db["bmkg-hw"].insert_many(forecast_docs)
        print(f"✓ Inserted {len(inserted.inserted_ids)} forecast records")
    else:
        print("⚠️ No forecast data inserted.")
    
    # Insert forecast documents
    # insert_result = db["bmkg-hw"].insert_many(forecast_docs)
    # print(f"✓ Inserted {len(insert_result.inserted_ids)} forecast documents into 'bmkg-hw' collection")
    
    # Summary
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Forecast period: {df.index[-1].strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')}")
    print(f"Total forecast days: {forecast_days}")
    print(f"Parameters forecasted: {', '.join(parameters)}")
    print("\nOptimal Parameters Found:")
    for param in parameters:
        params = results[param]["optimal_params"]
        print(f"{param}: α={params['alpha']}, β={params['beta']}, γ={params['gamma']}")
    
    client.close()
    print("\n✓ Analysis completed !")
    return forecast_docs


