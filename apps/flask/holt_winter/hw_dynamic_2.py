from pymongo import MongoClient
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
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
    if param_name in ["RR", "RR_imputed", "PRECTOTCORR"]: 
        if lam is not None:  
            forecast = (forecast * lam + 1) ** (1/lam) - 1
        if param_name == "PRECTOTCORR":
            forecast = np.clip(forecast, 0, 200)  
        else:
            forecast = np.clip(forecast, 0, 300)
    elif param_name == "NDVI":  
        forecast = np.clip(forecast, -1, 1) 
    elif param_name in ["RH_AVG", "RH_AVG_preprocessed", "RH2M"]:  
        forecast = np.clip(forecast, 0, 100)  
    elif param_name in ["TAVG", "TMAX", "TMIN", "T2M"]:  
        forecast = np.clip(forecast, 10, 50)  
    elif param_name in ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]:
        if param_name == "ALLSKY_SFC_SW_DWN":
            forecast = np.clip(forecast, 0, 30)
        else:
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


def grid_search_hw_params(train_data, param_name, validation_ratio=0.10):
    """
    Grid search disesuaikan untuk pola curah hujan Indonesia
    Return: best_params, error_metrics, predicted_validation
    """
    print(f"\n--- Grid Search for Indonesian Rainfall Pattern: {param_name} ---")
    print(f"📊 Using validation ratio: {validation_ratio * 100:.0f}%")
    
    is_ndvi = param_name in ["NDVI", "NDVI_imputed"]
    is_rainfall_or_radiation = param_name in ["RR", "RR_imputed", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]
    min_data_length = 46 if is_ndvi else 365

    if len(train_data) < min_data_length:
        print(f"❌ Insufficient data (need at least {min_data_length} {'pengukuran' if is_ndvi else 'hari'})")
        return None, None, None
    
    # Range lebih lebar untuk parameter yang noisy (curah hujan & radiasi)
    if is_rainfall_or_radiation:
        alpha_range = [0.1, 0.2, 0.3, 0.5, 0.7]
        beta_range = [0.01, 0.05, 0.1, 0.2, 0.3]
        gamma_range = [0.1, 0.2, 0.3, 0.5]
    else:
        alpha_range = [0.3, 0.5, 0.7]  
        beta_range = [0.1, 0.3, 0.5]
        gamma_range = [0.3, 0.5, 0.7]

    best_period = detect_seasonal_period(train_data, param_name)
    seasonal_periods_options = [best_period]
    if is_ndvi:
        seasonal_periods_options.extend([best_period//2, best_period*2] if best_period > 4 else [4])
    else:
        seasonal_periods_options.extend([best_period//2, best_period*2] if best_period > 7 else [7])
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    best_predicted_val = None
    valid_models = 0
    
    if is_ndvi:
        val_size = max(4, int(len(train_data) * validation_ratio))
    else:
        val_size = int(len(train_data) * validation_ratio)
        val_size = max(30, val_size) 

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
                        
                        forecast = model.forecast(len(val_split))
                        forecast = post_process_forecast(forecast, param_name)
                        
                        mae = mean_absolute_error(val_split, forecast)
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
                                'mape': mape,
                                'mse': mse,
                                'rmse': rmse
                            }
                            best_predicted_val = forecast
                            valid_models += 1
                            
                    except Exception:
                        continue
    
    if best_params:
        print(f"\n🎯 Best Params: {best_params}")
        print(f"📈 Metrics: {best_metrics}")
    
    return best_params, best_metrics, best_predicted_val


def save_historical_data_bulk(db, collection_name, target_column, train_data, val_data, 
                               predicted_val, split_name, config_id):
    """
    Menyimpan train, validation, dan predicted_validation ke hw-historical
    """
    print(f"\n💾 Saving historical data for {split_name}...")
    
    all_docs = []
    
    # Train documents
    for date, value in train_data.items():
        doc = {
            "date": date.to_pydatetime(),
            "timestamp": datetime.now().isoformat(),
            "source_collection": collection_name,
            "config_id": config_id,
            "split_ratio": split_name,
            "data_type": "train",
            "parameters": {
                target_column: {
                    "actual_value": float(value)
                }
            }
        }
        all_docs.append(doc)
    
    # Validation documents (actual)
    for date, value in val_data.items():
        doc = {
            "date": date.to_pydatetime(),
            "timestamp": datetime.now().isoformat(),
            "source_collection": collection_name,
            "config_id": config_id,
            "split_ratio": split_name,
            "data_type": "validation",
            "parameters": {
                target_column: {
                    "actual_value": float(value)
                }
            }
        }
        all_docs.append(doc)
    
    # Predicted validation documents
    if predicted_val is not None:
        for i, (date, actual_value) in enumerate(val_data.items()):
            if i < len(predicted_val):
                doc = {
                    "date": date.to_pydatetime(),
                    "timestamp": datetime.now().isoformat(),
                    "source_collection": collection_name,
                    "config_id": config_id,
                    "split_ratio": split_name,
                    "data_type": "predicted_validation",
                    "parameters": {
                        target_column: {
                            "predicted_value": float(predicted_val[i])
                        }
                    }
                }
                all_docs.append(doc)
    
    # Bulk insert
    if all_docs:
        db["hw-historical"].insert_many(all_docs, ordered=False)
        print(f"✅ Saved {len(all_docs)} historical documents")
    
    return len(all_docs)


def run_optimized_hw_analysis(collection_name, target_column, config_id=None, client=None, 
                               start_date=None, end_date=None):
    """
    Fungsi Holt-Winter yang dinamis berdasarkan parameter dari forecast_config
    Menjalankan 3 split ratio: 70:30, 80:20, 90:10
    Menyimpan ke hw-historical (train + validation + predicted_validation) dan holt-winter (forecast)
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
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"Fetched {len(source_data)} records from {collection_name}")
        
        if not source_data:
            raise ValueError(f"No data found in collection {collection_name}")
        
        df = pd.DataFrame(source_data)
        
        date_column = None
        for col in ['Date', 'date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            raise ValueError(f"No date column found in {collection_name}")
        
        df['timestamp'] = pd.to_datetime(df[date_column])
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        is_ndvi = target_column in ["NDVI", "NDVI_imputed"]
        freq = '16D' if is_ndvi else 'D'

        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        df = df.reindex(date_range)

        if target_column in ["PRECTOTCORR", "RR", "RR_imputed"]:
            df[target_column] = df[target_column].fillna(0)
        elif is_ndvi:
            df[target_column] = df[target_column].interpolate(method="linear", limit_direction="both")
        else:
            df[target_column] = df[target_column].interpolate(method="time", limit_direction="both")
        
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        param_data = df[target_column].dropna()
        lam = None
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data for {target_column}")
        
        print(f"Data summary: Total={len(param_data)}, Mean={param_data.mean():.3f}")

        # Decomposition
        try:
            best_period = detect_seasonal_period(param_data, target_column)
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
            print(f"✅ Decompose success: {len(decompose_docs)} documents")
            
        except Exception as e:
            print(f"❌ Decompose failed: {str(e)}")
        
        split_ratios = [
            {"ratio": 0.30, "name": "70:30"},
            {"ratio": 0.20, "name": "80:20"},
            {"ratio": 0.10, "name": "90:10"}
        ]
        
        all_results = []
        
        for split_config in split_ratios:
            val_ratio = split_config["ratio"]
            split_name = split_config["name"]
            
            print(f"\n{'='*60}")
            print(f"🔄 Processing split ratio: {split_name}")
            print(f"{'='*60}")
            
            # Split data
            if is_ndvi:
                val_size = max(4, int(len(param_data) * val_ratio))
            else:
                val_size = max(30, int(len(param_data) * val_ratio))
            
            split_point = len(param_data) - val_size
            train_split = param_data[:split_point]
            val_split = param_data[split_point:]
            
            print(f"📊 Train: {len(train_split)}, Validation: {len(val_split)}")
            
            # Grid search - RETURN predicted_validation
            best_params, error_metrics, predicted_val = grid_search_hw_params(
                param_data, target_column, validation_ratio=val_ratio
            )
            
            if best_params is None:
                print(f"⚠️  No valid model found for split {split_name}")
                continue
            
            # Save historical data (train + validation + predicted_validation)
            save_historical_data_bulk(
                db=db,
                collection_name=collection_name,
                target_column=target_column,
                train_data=train_split,
                val_data=val_split,
                predicted_val=predicted_val,
                split_name=split_name,
                config_id=config_id
            )
            
            # Fit final model dengan FULL data untuk forecast
            final_model = fit_robust_model(param_data, best_params, target_column)
            
            if final_model is None:
                print(f"⚠️  Failed to fit final model")
                continue
            
            # Forecast dates
            if start_date is not None and end_date is not None:
                forecast_start_date = pd.to_datetime(start_date)
                forecast_end_date = pd.to_datetime(end_date)
            else:
                last_data_date = df.index[-1]
                forecast_start_date = last_data_date + pd.Timedelta(days=1)
                forecast_end_date = forecast_start_date + pd.Timedelta(days=364)
        
            date_increment = '16D' if is_ndvi else 'D'
            forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq=date_increment)
            forecast_steps = len(forecast_dates)
            
            # Generate forecast
            try:
                forecast = final_model.forecast(steps=forecast_steps)
                
                if hasattr(forecast, 'values'):
                    forecast = forecast.values
                
                forecast = np.array(forecast).flatten()
                forecast = post_process_forecast(forecast, target_column, lam)
                
                print(f"✓ Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
                
            except Exception as e:
                raise ValueError(f"Forecast generation failed: {str(e)}")
            
            # Save forecast documents
            forecast_docs = []
            
            for i, forecast_date in enumerate(forecast_dates):
                forecast_value = float(forecast[i])
                
                if np.isnan(forecast_value) or np.isinf(forecast_value):
                    continue

                doc = {
                    "forecast_date": forecast_date.to_pydatetime(),
                    "timestamp": datetime.now().isoformat(),
                    "source_collection": collection_name,
                    "config_id": config_id,
                    "split_ratio": split_name,
                    "parameters": {
                        target_column: {
                            "forecast_value": forecast_value,
                            "model_metadata": {
                                "alpha": best_params["alpha"],
                                "beta": best_params["beta"],
                                "gamma": best_params["gamma"],
                                "seasonal_periods": best_params.get("seasonal_periods", 23 if is_ndvi else 7)
                            }
                        }
                    }
                }
                forecast_docs.append(doc)
            
            if forecast_docs:
                db["holt-winter"].insert_many(forecast_docs, ordered=False)
                print(f"✅ Saved {len(forecast_docs)} forecast documents")
            
            result_summary = {
                "collection_name": collection_name,
                "target_column": target_column,
                "split_ratio": split_name,
                "train_days": len(train_split),
                "validation_days": len(val_split),
                "forecast_days": len(forecast_docs),
                "model_params": best_params,
                "error_metrics": error_metrics
            }
            
            all_results.append(result_summary)
        
        return all_results if all_results else [{"error": "No valid models found"}]
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise e
    
    finally:
        if should_close_client:
            client.close()

if __name__ == "__main__":
    run_optimized_hw_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )