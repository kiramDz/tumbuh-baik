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
    """
    Perform grid search for optimal Holt-Winters parameters
    """
    print(f"\n--- Grid Search for {param_name} ---")
    print(f"Data shape: {len(train_data)} points")
    print(f"Data range: {train_data.min():.3f} to {train_data.max():.3f}")
    print(f"Zero values: {(train_data == 0).sum()} ({(train_data == 0).mean()*100:.1f}%)")
    
    # Define parameter grid - lebih fleksibel
    alpha_range = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    beta_range = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_range = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    best_score = float('inf')
    best_params = None
    best_model = None
    valid_models = 0
    
    # Split data for validation - lebih besar untuk stability
    train_split, val_split = time_series_split(train_data, test_size=0.2)
    
    print(f"Train split: {len(train_split)} points")
    print(f"Validation split: {len(val_split)} points")
    
    total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range)
    current_combination = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Coba beberapa seasonal periods
    seasonal_periods_options = [365, 30, 7]  # tahunan, bulanan, mingguan
    
    for seasonal_periods in seasonal_periods_options:
        print(f"\nTrying seasonal_periods = {seasonal_periods}")
        
        for alpha, beta, gamma in itertools.product(alpha_range, beta_range, gamma_range):
            current_combination += 1
            
            try:
                # Fit model with current parameters
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
                
                # Forecast for validation period
                forecast = model.forecast(steps=len(val_split))
                
                # Calculate error metrics
                mae = mean_absolute_error(val_split, forecast)
                rmse = np.sqrt(mean_squared_error(val_split, forecast))
                
                # Skip if forecast contains NaN or infinite values
                if np.isnan(forecast).any() or np.isinf(forecast).any():
                    continue
                
                # Calculate MAPE carefully (avoid division by zero)
                non_zero_mask = val_split != 0
                if non_zero_mask.sum() > 0:
                    mape = calculate_mape(val_split[non_zero_mask], forecast[non_zero_mask])
                else:
                    mape = float('inf')
                
                # Use RMSE as primary metric
                score = rmse
                valid_models += 1
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'alpha': alpha, 
                        'beta': beta, 
                        'gamma': gamma,
                        'seasonal_periods': seasonal_periods
                    }
                    best_model = model
                    print(f"✓ New best: α={alpha}, β={beta}, γ={gamma}, SP={seasonal_periods} | RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%")
            
            except Exception as e:
                # Skip problematic parameter combinations
                continue
            
            # Progress indicator
            if current_combination % 50 == 0:
                progress = (current_combination / (total_combinations * len(seasonal_periods_options))) * 100
                print(f"Progress: {current_combination}/{total_combinations * len(seasonal_periods_options)} ({progress:.1f}%) | Valid models: {valid_models}")
        
        # If we found good parameters, break early
        if best_params is not None:
            break
    
    if best_params is None:
        print(f"❌ No valid parameter combination found for {param_name} after testing {current_combination} combinations.")
        print("This might be due to:")
        print("- Too many zero/missing values in the data")
        print("- Insufficient seasonal patterns")
        print("- Data quality issues")
        return None, None
    
    print(f"\n✓ Final best parameters for {param_name}:")
    print(f"Alpha (level): {best_params['alpha']}")
    print(f"Beta (trend): {best_params['beta']}")
    print(f"Gamma (seasonal): {best_params['gamma']}")
    print(f"Seasonal periods: {best_params['seasonal_periods']}")
    print(f"Best RMSE: {best_score:.3f}")
    print(f"Valid models found: {valid_models}")
    
    return best_params, best_model

def calculate_forecast_horizon(start_date, end_date):
    """Calculate number of days between two dates"""
    return (end_date - start_date).days

def run_optimized_hw_analysis():
    print("=== Start Optimized Holt-Winter Analysis ===")
    
    # Connect to MongoDB
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
        
        # Check if grid search was successful
        if best_params is None:
            print(f"⚠️ Grid search failed for {param}. Using fallback approach...")
            
            # Fallback: Try simple exponential smoothing or basic parameters
            try:
                # Try without seasonal component first
                fallback_model = ExponentialSmoothing(
                    param_data,
                    trend="add",
                    seasonal=None
                ).fit()
                
                best_params = {
                    'alpha': 0.3,
                    'beta': 0.1, 
                    'gamma': 0.1,
                    'seasonal_periods': 365,
                    'use_seasonal': False
                }
                print(f"✓ Using fallback parameters for {param} (no seasonal)")
                
            except Exception as fallback_error:
                print(f"❌ Fallback also failed for {param}: {str(fallback_error)}")
                continue
        
        # Fit final model with best parameters on full dataset
        print(f"\nFitting final model for {param} with optimal parameters...")
        
        try:
            if best_params.get('use_seasonal', True):
                final_model = ExponentialSmoothing(
                    param_data,
                    trend="add",
                    seasonal="add", 
                    seasonal_periods=best_params.get('seasonal_periods', 365)
                ).fit(
                    smoothing_level=best_params['alpha'],
                    smoothing_trend=best_params['beta'],
                    smoothing_seasonal=best_params['gamma'],
                    optimized=False
                )
            else:
                # No seasonal component
                final_model = ExponentialSmoothing(
                    param_data,
                    trend="add",
                    seasonal=None
                ).fit(
                    smoothing_level=best_params['alpha'],
                    smoothing_trend=best_params['beta'],
                    optimized=False
                )
        except Exception as model_error:
            print(f"❌ Failed to fit final model for {param}: {str(model_error)}")
            continue
        
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
    
    # Insert forecast documents
    insert_result = db["bmkg-hw"].insert_many(forecast_docs)
    print(f"✓ Inserted {len(insert_result.inserted_ids)} forecast documents into 'bmkg-hw' collection")
    
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
    print("\n✓ Analysis completed successfully!")
    return forecast_docs

if __name__ == "__main__":
    run_optimized_hw_analysis()