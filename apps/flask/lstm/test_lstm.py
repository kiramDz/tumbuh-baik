from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import itertools
import warnings

warnings.filterwarnings('ignore')

# ======================================================
# Fungsi post-process hasil forecast - OPTIMIZED
# ======================================================
def post_process_forecast(forecast, param_name):
    """Optimized dengan numpy vectorization"""
    if param_name in ["RR", "RR_imputed"]:
        return np.clip(forecast, 0, 300)
    elif param_name == "RH_AVG":
        return np.clip(forecast, 0, 100)
    elif param_name == "NDVI":
        return np.clip(forecast, -1, 1)
    elif "Suhu" in param_name or "Temperature" in param_name:
        return np.clip(forecast, -50, 60)
    return forecast

# ======================================================
# Dataset supervised (X,y) untuk LSTM - OPTIMIZED
# ======================================================
def create_supervised_data(data_scaled, lookback):
    """Optimized dengan numpy advanced indexing - menghilangkan loop Python"""
    if len(data_scaled) <= lookback:
        return np.array([]), np.array([])
    
    n_samples = len(data_scaled) - lookback
    # Vectorized approach - jauh lebih cepat dari loop Python
    indices = np.arange(lookback)[None, :] + np.arange(n_samples)[:, None]
    
    X = data_scaled[indices].reshape(n_samples, lookback, 1)
    y = data_scaled[lookback:]
    
    return X, y

# ======================================================
# Model LSTM
# ======================================================
def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ======================================================
# Recursive Forecast - OPTIMIZED
# ======================================================
def lstm_recursive_forecast(model, last_window, horizon):
    """Optimized dengan numpy operations yang lebih efisien"""
    predictions = np.zeros(horizon)
    current_window = last_window.copy()
    
    for i in range(horizon):
        pred = model.predict(current_window[np.newaxis, :, :], verbose=0)[0, 0]
        predictions[i] = pred
        
        # Optimized window update - lebih cepat dari np.vstack
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred
    
    return predictions

# ======================================================
# Validation function untuk mengurangi redundant code
# ======================================================
def validate_single_combination(train_split, val_split, lookback, scaler_type, param_name):
    """Consolidated validation untuk menghindari kode berulang"""
    try:
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        train_scaled = scaler.fit_transform(train_split.values.reshape(-1, 1))
        val_scaled = scaler.transform(val_split.values.reshape(-1, 1))

        X_train, y_train = create_supervised_data(train_scaled, lookback)
        X_val, y_val = create_supervised_data(val_scaled, lookback)
        
        if len(X_train) == 0 or len(X_val) == 0:
            return None

        model = build_lstm_model(input_shape=(lookback, 1))
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True  # Reduced patience untuk epochs rendah
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # ✅ CHANGED: Reduced dari 50 ke 10
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        val_pred_scaled = model.predict(X_val, verbose=0).flatten()
        val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
        val_true = val_split.iloc[lookback:].values

        if len(val_pred) != len(val_true):
            return None

        val_pred = post_process_forecast(val_pred, param_name)
        if np.isnan(val_pred).any() or np.isinf(val_pred).any():
            return None

        mse = mean_squared_error(val_true, val_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true, val_pred)
        mape = np.mean(np.abs((val_true - val_pred) / np.where(val_true != 0, val_true, 1))) * 100

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mse': mse,
            'lookback': lookback,
            'scaler_type': scaler_type
        }
        
    except Exception:
        return None

# ======================================================
# Grid search parameter LSTM - OPTIMIZED
# ======================================================
def grid_search_lstm_params(train_data, param_name):
    """Optimized grid search dengan reduced redundancy"""
    print(f"\n--- LSTM Grid Search for {param_name} (FAST MODE: 10 epochs) ---")

    if len(train_data) < 100:
        print("❌ Data terlalu sedikit untuk grid search")
        return None, None

    # Optimized lookback untuk data 20 tahun
    if len(train_data) > 5000:  # Data 20 tahun (~7300 hari)
        lookback_range = [365, 730, 1095, 1460, 1825]  # 1-5 tahun
        print(f"📅 Data span: {len(train_data)} days (~{len(train_data)/365:.1f} years)")
    elif len(train_data) > 2000:
        lookback_range = [365, 730, 1095]  # 1-3 tahun
    else:
        lookback_range = [180, 365, 700]
    
    scaler_range = ['standard', 'minmax']

    split_point = int(len(train_data) * 0.8)  # Lebih banyak data training
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    print(f"📊 Train: {len(train_split)} days, Val: {len(val_split)} days")

    # Pre-filter valid lookbacks untuk menghindari checking berulang
    valid_lookbacks = [lb for lb in lookback_range if len(train_split) > lb + 100]
    
    if not valid_lookbacks:
        print("❌ No valid lookback values for this dataset")
        return None, None

    print(f"🔍 Testing {len(valid_lookbacks) * len(scaler_range)} combinations (10 epochs each)...")

    best_score = float('inf')
    best_result = None
    valid_models = 0

    # Optimized grid search loop
    for lookback, scaler_type in itertools.product(valid_lookbacks, scaler_range):
        result = validate_single_combination(train_split, val_split, lookback, scaler_type, param_name)
        
        if result is not None:
            valid_models += 1
            if result['rmse'] < best_score:
                best_score = result['rmse']
                best_result = result
                print(f"✅ New best: lookback={lookback} days ({lookback/365:.1f}y), scaler={scaler_type}")
                print(f"   RMSE={result['rmse']:.3f}, MAE={result['mae']:.3f}, MAPE={result['mape']:.1f}%")

    if best_result is None:
        print("❌ Tidak ada model LSTM yang valid")
        return None, None

    # Format hasil terbaik
    best_params = {
        'lookback': best_result['lookback'],
        'lookback_years': round(best_result['lookback']/365, 2),
        'units': 128,
        'dropout': 0.2,
        'scaler_type': best_result['scaler_type'],
        'epochs': 10,  # ✅ CHANGED: Reduced dari 150 ke 10
        'batch_size': 64
    }

    error_metrics = {
        'mae': best_result['mae'],
        'rmse': best_result['rmse'], 
        'mape': best_result['mape'],
        'mse': best_result['mse']
    }

    print(f"\n🎯 BEST parameters for {param_name} (FAST MODE):")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    print(f"✅ Valid models tested: {valid_models}")

    return best_params, error_metrics

# ======================================================
# Training final model - OPTIMIZED
# ======================================================
def fit_final_lstm_model(data, best_params, param_name):
    try:
        print(f"\n🚀 Training final LSTM model for {param_name} (FAST MODE)")
        print(f"📅 Lookback: {best_params['lookback']} days ({best_params.get('lookback_years', 0)} years)")
        print(f"⚡ Epochs: {best_params['epochs']} (fast training)")
        
        scaler = StandardScaler() if best_params['scaler_type'] == 'standard' else MinMaxScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = create_supervised_data(data_scaled, best_params['lookback'])
        if len(X) == 0:
            raise ValueError("Insufficient data after creating sequences")

        print(f"📊 Total samples: {len(X)}")
        
        # Untuk data 20 tahun, gunakan 90:10 split
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📈 Final train: {len(X_train)}, validation: {len(X_test)}")

        model = build_lstm_model(input_shape=(best_params['lookback'], 1))
        
        # Simplified callbacks untuk training cepat
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5,  # ✅ CHANGED: Reduced patience dari 25 ke 5
                restore_best_weights=True,
                min_delta=0.001  # Increased tolerance
            )
        ]

        print(f"🎯 Starting fast training (10 epochs)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test) if len(X_test) > 0 else None,
            epochs=best_params.get('epochs', 10),  # ✅ CHANGED: 10 epochs
            batch_size=best_params.get('batch_size', 64),
            callbacks=callbacks,
            verbose=1
        )

        # Training analysis
        if len(X_test) > 0 and 'val_loss' in history.history:
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            gap_percent = abs(final_val_loss - final_train_loss) / final_train_loss * 100
            
            print(f"\n📊 Fast training completed:")
            print(f"   Train Loss: {final_train_loss:.6f}")
            print(f"   Val Loss: {final_val_loss:.6f}")
            print(f"   Gap: {gap_percent:.2f}%")
            print(f"   Epochs trained: {len(history.history['loss'])}")
            
            if gap_percent > 30:  # More tolerant untuk fast training
                print("⚠️  Note: Limited training with 10 epochs")
            else:
                print("✅ Good: Reasonable performance for fast training")

        return model, scaler
        
    except Exception as e:
        print(f"❌ Final model fitting failed: {e}")
        return None, None

# ======================================================
# Pipeline utama - OPTIMIZED untuk FAST MODE
# ======================================================
def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast",
                       config_id=None, append_column_id=True, client=None):
    print(f"=== 🚀 Start FAST LSTM Analysis for {collection_name}.{target_column} ===")
    print(f"⚡ Mode: 10 epochs, 20-day forecast")

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
        # Fetch data
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"📥 Fetched {len(source_data)} records from {collection_name}")
        if not source_data:
            raise ValueError(f"No data found in collection {collection_name}")

        # Prepare DataFrame
        df = pd.DataFrame(source_data)
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")

        # Clean data
        param_data = pd.to_numeric(df[target_column], errors='coerce').dropna()
        if len(param_data) < 200:
            raise ValueError(f"Insufficient data for LSTM {target_column} (need ≥200 points)")

        print(f"📊 Usable data points: {len(param_data)} (~{len(param_data)/365:.1f} years)")

        # Grid search
        best_params, error_metrics = grid_search_lstm_params(param_data, target_column)
        if best_params is None:
            raise ValueError(f"No valid LSTM model found for {target_column}")

        # Train final model
        final_model, scaler = fit_final_lstm_model(param_data, best_params, target_column)
        if final_model is None or scaler is None:
            raise ValueError(f"Failed to fit final LSTM model for {target_column}")

        # Calculate forecast horizon - ✅ CHANGED: 20 hari instead of 1 year
        data_end_date = pd.to_datetime(df["Date"].iloc[-1])
        forecast_days = 20  # ✅ CHANGED: Fixed 20 days forecast
        print(f"📅 Forecast horizon: {forecast_days} days (20-day forecast)")

        # Generate forecast
        print(f"🔮 Generating LSTM forecast for {target_column} (20 days)...")
        data_scaled = scaler.transform(param_data.values.reshape(-1, 1))
        last_window = data_scaled[-best_params['lookback']:]
        
        forecast_scaled = lstm_recursive_forecast(final_model, last_window, forecast_days)
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        forecast = post_process_forecast(forecast, target_column)

        print(f"✅ {target_column} LSTM forecast completed (20 days)")
        print(f"📈 Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")

        # Optimized document creation - batch processing
        forecast_docs = []
        base_date = data_end_date
        
        # Vectorized date generation - ✅ CHANGED: 20 days only
        forecast_dates = pd.date_range(
            start=base_date + pd.Timedelta(days=1), 
            periods=forecast_days,  # 20 days
            freq='D'
        )
        
        for i, (forecast_date, forecast_value) in enumerate(zip(forecast_dates, forecast)):
            if np.isnan(forecast_value) or np.isinf(forecast_value):
                continue

            doc = {
                "forecast_date": forecast_date.to_pydatetime(),
                "timestamp": datetime.now().isoformat(),
                "source_collection": collection_name,
                "config_id": config_id,
                "parameters": {
                    target_column: {
                        "forecast_value": float(forecast_value),
                        "model_metadata": {
                            "model": "LSTM_FAST",  # ✅ CHANGED: Indicate fast mode
                            "mode": "fast_training",
                            "lookback_days": best_params['lookback'],
                            "lookback_years": best_params.get('lookback_years', 0),
                            "scaler": best_params['scaler_type'],
                            "units": best_params['units'],
                            "dropout": best_params['dropout'],
                            "epochs": best_params.get('epochs', 10),  # 10 epochs
                            "batch_size": best_params.get('batch_size', 64),
                            "forecast_horizon_days": forecast_days  # 20 days
                        }
                    }
                }
            }
            if append_column_id:
                doc["column_id"] = f"{collection_name}_{target_column}"
            forecast_docs.append(doc)

        # Database upsert
        print(f"💾 Saving {len(forecast_docs)} forecast documents...")
        upsert_count = 0
        for doc in forecast_docs:
            result = db[save_collection].update_one(
                {"forecast_date": doc["forecast_date"], "config_id": config_id},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1

        print(f"✅ Saved {upsert_count} forecast documents to {save_collection}")

        return {
            "collection_name": collection_name,
            "target_column": target_column,
            "forecast_days": len(forecast_docs),
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_params": best_params,
            "error_metrics": error_metrics,
            "forecast_range": {
                "min": float(forecast.min()) if len(forecast) > 0 else None,
                "max": float(forecast.max()) if len(forecast) > 0 else None
            },
            "data_years": round(len(param_data)/365, 1),
            "training_mode": "fast",
            "epochs_used": best_params.get('epochs', 10),
            "forecast_horizon_days": forecast_days
        }

    except Exception as e:
        print(f"❌ Error in FAST LSTM analysis: {str(e)}")
        raise e

    finally:
        if should_close_client:
            client.close()

# ======================================================
# Eksekusi langsung
# ======================================================
if _name_ == "_main_":
    run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )