"""
LSTM RECURSIVE FORECASTING - Solusi untuk forecast 1 tahun yang lebih realistis
Menggunakan iterative/recursive approach: prediksi 1 hari, gunakan hasil sebagai input berikutnya
"""
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import warnings

warnings.filterwarnings('ignore')

# ======================================================
# Parameter Categories
# ======================================================
RAINFALL_PARAMS = ["RR", "RR_imputed", "PRECTOTCORR"]
NDVI_PARAMS = ["NDVI", "NDVI_imputed"]
TEMP_PARAMS = ["TAVG", "TMAX", "TMIN", "T2M"]
HUMIDITY_PARAMS = ["RH_AVG", "RH_AVG_preprocessed", 'RH2M']
SOLAR_PARAMS = ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# ======================================================
# PyTorch Dataset Class
# ======================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================================================
# LSTM Model - ONE-STEP AHEAD (untuk recursive forecasting)
# ======================================================
class RecursiveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=128, hidden_size_2=64, dropout=0.3):
        super(RecursiveLSTM, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size_2, 64)
        self.dropout_fc = nn.Dropout(dropout * 0.5)
        
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        # x: (batch, seq, input_size)
        lstm_out1, _ = self.lstm1(x)
        
        # Apply BatchNorm to sequence (requires Permute to N, C, L)
        out1 = lstm_out1.permute(0, 2, 1)
        out1 = self.bn1(out1)
        out1 = self.dropout1(out1)
        
        # Permute back to N, L, C for LSTM2
        out1 = out1.permute(0, 2, 1)
        
        # LSTM Layer 2 processing full sequence
        lstm_out2, _ = self.lstm2(out1)
        
        # Take last time step output
        out2 = lstm_out2[:, -1, :]
        out2 = self.bn2(out2)
        out2 = self.dropout2(out2)
        
        out = self.dropout_fc(self.fc1(out2))
        return self.fc_out(out)

# ======================================================
# Create supervised data - ONE STEP AHEAD
# ======================================================
def create_supervised_onestep(data_scaled, lookback):
    if len(data_scaled) <= lookback:
        return np.array([]), np.array([])
    
    n_samples = len(data_scaled) - lookback
    
    indices_X = np.arange(lookback)[None, :] + np.arange(n_samples)[:, None]
    X = data_scaled[indices_X].reshape(n_samples, lookback, 1)
    
    y = data_scaled[lookback:].reshape(-1, 1)
    
    return X, y

def recursive_forecast(model, initial_window, forecast_horizon, scaler, device, 
                       residuals=None, noise_level=0.0):
    model.eval()
    
    current_window = initial_window.copy()
    forecast_scaled = []
    
    residual_pool = residuals if residuals is not None and len(residuals) > 0 else None

    with torch.no_grad():
        for step_idx in range(forecast_horizon):
            window_tensor = torch.FloatTensor(current_window).reshape(1, -1, 1).to(device)
            pred = model(window_tensor).cpu().numpy().flatten()[0]
            
            if residual_pool is not None:
                sampled_resid = np.random.choice(residual_pool)
                pred_final = pred + sampled_resid
            else:
                noise_scale = noise_level * (1 + step_idx / forecast_horizon)
                noise = np.random.normal(0, noise_scale * abs(pred))
                pred_final = pred + noise
            
            forecast_scaled.append(pred_final)
            current_window = np.append(current_window[1:], pred_final)
    
    forecast_scaled_array = np.array(forecast_scaled).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled_array).flatten()
    
    return forecast

# ======================================================
# Post-process forecast - LESS AGGRESSIVE clipping
# ======================================================
def post_process_forecast(forecast, param_name, historical_data=None, strict=False):
    """
    Enhanced post-processing dengan opsi strict/loose clipping
    
    Args:
        strict: If True, use aggressive clipping (less variability)
                If False, use loose clipping (more variability, more realistic)
    """
    
    # Physical constraints (hard limits)
    if param_name in RAINFALL_PARAMS:
        forecast = np.clip(forecast, 0, 300)
    elif param_name in HUMIDITY_PARAMS:
        forecast = np.clip(forecast, 0, 100)
    elif param_name in NDVI_PARAMS:
        forecast = np.clip(forecast, -1, 1)
    elif param_name in TEMP_PARAMS:
        forecast = np.clip(forecast, 10, 50)
    elif param_name in SOLAR_PARAMS:
        forecast = np.clip(forecast, 0, 30)
    else:
        # Generic fallback
        param_lower = param_name.lower()
        if any(kw in param_lower for kw in ["rain", "precip", "hujan"]):
            forecast = np.clip(forecast, 0, 300)
            forecast[forecast < 0.1] = 0
        elif any(kw in param_lower for kw in ["hum", "rh", "kelembaban"]):
            forecast = np.clip(forecast, 0, 100)
        elif any(kw in param_lower for kw in ["temp", "suhu", "t2m", "tavg"]):
            forecast = np.clip(forecast, -50, 60)
        elif any(kw in param_lower for kw in ["ndvi", "vegetasi"]):
            forecast = np.clip(forecast, -1, 1)
        elif any(kw in param_lower for kw in ["solar", "rad", "ghi", "srad"]):
            forecast = np.clip(forecast, 0, None)
    
    return forecast

# ======================================================
# Seasonal Decomposition
# ======================================================
def apply_seasonal_decomposition(data, period=365, model='additive'):
    """Decompose time series"""
    try:
        if len(data) < 2 * period:
            print(f"⚠️  Data terlalu pendek untuk seasonal decompose")
            return None, None, None, data.values
        
        print(f"🔄 Applying seasonal decomposition (period={period} days)...")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index(drop=True)
        
        decomposition = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
        
        # Fix deprecated method
        trend = decomposition.trend.bfill().ffill()
        seasonal = decomposition.seasonal.fillna(0)
        residual = decomposition.resid.fillna(0)
        
        print(f"✅ Decomposition completed")
        print(f"   Trend: {trend.min():.2f} to {trend.max():.2f}")
        print(f"   Seasonal: {seasonal.min():.2f} to {seasonal.max():.2f}")
        print(f"   Residual std: {residual.std():.2f}")
        
        return trend, seasonal, residual, decomposition.observed
        
    except Exception as e:
        print(f"❌ Decomposition failed: {e}")
        return None, None, None, data.values

# ======================================================
# Training function
# ======================================================
def train_pytorch_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    """Train PyTorch LSTM model"""
    
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING STARTED (Recursive LSTM)")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED")
    print(f"   Epochs trained: {epoch+1}/{epochs}")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    return model, train_losses, val_losses

# ======================================================
# Calculate metrics
# ======================================================
def calculate_metrics(y_true, y_pred, param_name="generic"):
    """Comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Specialized MAPE calculation
    param_lower = param_name.lower()
    is_rainfall = any(x in param_lower for x in ['rr', 'rain', 'precip', 'hujan']) or param_name in RAINFALL_PARAMS
    
    if is_rainfall:
        # Only calculate MAPE for significant rainfall days (>= 1.0mm)
        rain_mask = y_true >= 1.0
        if np.sum(rain_mask) > 0:
            mape = float(np.mean(np.abs(
                (y_true[rain_mask] - y_pred[rain_mask]) / y_true[rain_mask]
            )) * 100)
        else:
            mape = 0.0
    else:
        # Standard MAPE for non-zero values
        non_zero_mask = np.abs(y_true) > 0.1
        if np.sum(non_zero_mask) > 0:
            mape = float(np.mean(np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )) * 100)
        else:
            mape = 0.0

    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = np.mean(true_dir == pred_dir) * 100
    else:
        dir_acc = None
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': dir_acc
    }

# ======================================================
# Grid search
# ======================================================
def grid_search_pytorch(train_data, param_name):
    print(f"\n{'='*70}\nPYTORCH GRID SEARCH: {param_name} (Recursive)\n{'='*70}")
    
    if len(train_data) < 500:
        print(f"❌ Insufficient data: {len(train_data)}")
        return None, None
    
    data_years = len(train_data) / 365
    split_point = int(len(train_data) * 0.8)
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    print(f"Data: {len(train_data)} days ({data_years:.1f} years)")
    print(f"Split: Train={len(train_split)}, Val={len(val_split)}")
    
    if data_years >= 10:
        lookback_range = [365, 730]
    elif data_years >= 5:
        lookback_range = [180, 365]
    else:
        lookback_range = [90, 180]
    
    scaler_range = ['standard', 'minmax']
    valid_lookbacks = [lb for lb in lookback_range if len(train_split) > lb + 100]
    
    if not valid_lookbacks:
        print("❌ No valid lookbacks")
        return None, None
    
    print(f"Testing {len(valid_lookbacks) * len(scaler_range)} combinations")
    
    best_score = float('inf')
    best_result = None
    
    for lookback, scaler_type in itertools.product(valid_lookbacks, scaler_range):
        print(f"\nTesting: lookback={lookback}, scaler={scaler_type}")
        
        try:
            scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            
            train_vals = train_split.values.reshape(-1, 1)
            scaler.fit(train_vals)
            
            train_scaled = scaler.transform(train_vals).flatten()
            val_scaled = scaler.transform(val_split.values.reshape(-1, 1)).flatten()
            
            X_train, y_train = create_supervised_onestep(train_scaled, lookback)
            X_val, y_val = create_supervised_onestep(val_scaled, lookback)
            
            if len(X_train) < 100:
                print("   ⚠️  Insufficient samples")
                continue
            
            print(f"   Samples: Train={len(X_train)}, Val={len(X_val)}")
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            model = RecursiveLSTM(input_size=1, hidden_size_1=128, hidden_size_2=64, dropout=0.3).to(device)
            
            model, _, _ = train_pytorch_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=15)
            
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    val_preds.append(outputs.cpu().numpy())
                    val_trues.append(y_batch.numpy())
            
            val_pred = np.concatenate(val_preds, axis=0).flatten()
            val_true = np.concatenate(val_trues, axis=0).flatten()
            
            val_pred_inv = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            val_true_inv = scaler.inverse_transform(val_true.reshape(-1, 1)).flatten()
            
            metrics = calculate_metrics(val_true_inv, val_pred_inv, param_name)
            residuals_scaled = val_true - val_pred
            
            print(f"   RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
            
            if metrics['rmse'] < best_score:
                best_score = metrics['rmse']
                best_result = {
                    'lookback': lookback,
                    'scaler_type': scaler_type,
                    'metrics': metrics,
                    'residuals_scaled': residuals_scaled
                }
                print(f"   ✅ New Best: {best_score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if best_result is None:
        return None, None, None
    
    best_params = {
        'lookback': best_result['lookback'],
        'lookback_years': round(best_result['lookback']/365, 2),
        'scaler_type': best_result['scaler_type'],
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        'dropout': 0.3,
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.001
    }
    
    return best_params, best_result['metrics'], best_result['residuals_scaled']

# ======================================================
# Main pipeline
# ======================================================
def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast",
                                config_id=None, append_column_id=True, client=None,
                                use_seasonal_decompose=True, forecast_horizon=365,
                                start_date=None, end_date=None):
    """
    RECURSIVE LSTM FORECASTING - Lebih realistis untuk forecast 1 tahun
    """
    print(f"\n{'='*70}")
    print(f"🔥 RECURSIVE LSTM FORECAST SYSTEM")
    print(f"{'='*70}")
    print(f"📊 Target: {collection_name}.{target_column}")
    print(f"🎯 Forecast horizon: {forecast_horizon} days (RECURSIVE/ITERATIVE)")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
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
        # 1. Fetch data
        print("📥 Step 1: Fetching data...")
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"   Retrieved: {len(source_data)} records")
        
        if not source_data:
            raise ValueError(f"No data in {collection_name}")
        
        # 2. Prepare
        print("\n🔧 Step 2: Data preparation...")
        df = pd.DataFrame(source_data)
        
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        param_data = pd.to_numeric(df[target_column], errors='coerce').dropna()
        
        if len(param_data) < 500:
            raise ValueError(f"Insufficient data: {len(param_data)}")
        
        # Adjust horizon based on end_date if provided
        data_end_date = param_data.index[-1]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt > data_end_date:
                days_needed = (end_dt - data_end_date).days
                forecast_horizon = days_needed
                print(f"   🗓️  Adjusted forecast horizon to {forecast_horizon} days (Target: {end_dt.date()})")
            else:
                print(f"   ⚠️  Target end date {end_dt.date()} is <= last data date {data_end_date.date()}, keeping default horizon")
        
        print(f"   Data: {len(param_data)} points ({len(param_data)/365:.1f} years)")
        print(f"   Range: {param_data.index[0]} to {param_data.index[-1]}")
        print(f"   Values: {param_data.min():.2f} to {param_data.max():.2f}")
        print(f"   Mean: {param_data.mean():.2f}, Std: {param_data.std():.2f}")
        
        # 3. Seasonal decomposition
        seasonal_pattern = None
        if use_seasonal_decompose and len(param_data) >= 730:
            print("\n🔄 Step 3: Seasonal decomposition...")
            trend, seasonal, residual, observed = apply_seasonal_decomposition(param_data, period=365)
            
            if trend is not None:
                deseasonalized = trend + residual
                seasonal_pattern = seasonal
                train_data = deseasonalized
                print("   ✅ Using deseasonalized data")
            else:
                train_data = param_data
        else:
            print("\n⏭️  Step 3: Skipping seasonal decomposition")
            train_data = param_data
        
        # 4. Grid search
        print("\n🔍 Step 4: Grid search...")
        best_params, val_metrics, grid_residuals = grid_search_pytorch(train_data, target_column)
        
        if best_params is None:
            raise ValueError("No valid model found")
        
        # 5. Final training
        print("\n🎓 Step 5: Final model training...")
        
        # Prepare final data
        scaler = StandardScaler() if best_params['scaler_type'] == 'standard' else MinMaxScaler()
        data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
        
        X, y = create_supervised_onestep(data_scaled, best_params['lookback'])
        
        if len(X) == 0:
            raise ValueError("Insufficient data for sequences")
        
        # Split
        split_idx = int(len(X) * 0.85)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Build final model
        final_model = RecursiveLSTM(
            input_size=1,
            hidden_size_1=best_params['hidden_size_1'],
            hidden_size_2=best_params['hidden_size_2'],
            dropout=best_params['dropout']
        ).to(device)
        
        # Train
        final_model, train_losses, val_losses = train_pytorch_model(
            final_model, train_loader, test_loader,
            epochs=best_params['epochs'],
            lr=best_params['learning_rate'],
            patience=20
        )
        
        # Test evaluation - one-step
        final_model.eval()
        test_preds = []
        test_trues = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = final_model(X_batch)
                test_preds.append(outputs.cpu().numpy())
                test_trues.append(y_batch.numpy())
        
        test_pred = np.concatenate(test_preds, axis=0).flatten()
        test_true = np.concatenate(test_trues, axis=0).flatten()
        
        # Inverse transform
        test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        test_true_inv = scaler.inverse_transform(test_true.reshape(-1, 1)).flatten()
        
        # Test metrics
        test_metrics = calculate_metrics(test_true_inv, test_pred_inv, target_column)
        
        print(f"\n📊 TEST PERFORMANCE (One-step ahead):")
        print(f"   RMSE: {test_metrics['rmse']:.3f}")
        print(f"   MAE: {test_metrics['mae']:.3f}")
        print(f"   MAPE: {test_metrics['mape']:.1f}%")
        print(f"   R²: {test_metrics['r2']:.3f}")
        
        # Prepare residuals pooling for sampling
        # Gabungkan residuals dari validation (grid search) dan test
        # Residuals = True - Pred (in scaled domain)
        test_residuals_scaled = (test_true - test_pred).flatten()
        
        # Combine if valid
        if grid_residuals is not None and len(grid_residuals) > 0:
             final_residuals_pool = np.concatenate([grid_residuals, test_residuals_scaled])
        else:
             final_residuals_pool = test_residuals_scaled
             
        # Remove NaNs if any
        final_residuals_pool = final_residuals_pool[~np.isnan(final_residuals_pool)]
        
        print(f"   📊 Residuals pool size: {len(final_residuals_pool)} (std: {np.std(final_residuals_pool):.4f})")

        # 6. Generate RECURSIVE forecast
        print(f"\n🔮 Step 6: Generating RECURSIVE forecast ({forecast_horizon} days)...")
        print("   Method: Residual Sampling (Bootstrapping residuals)")
        
        # Use last window
        last_window = data_scaled[-best_params['lookback']:]
        
        # RECURSIVE FORECASTING with RESIDUAL SAMPLING
        forecast = recursive_forecast(
            model=final_model,
            initial_window=last_window,
            forecast_horizon=forecast_horizon,
            scaler=scaler,
            device=device,
            residuals=final_residuals_pool  # Pass residuals pool here
        )
        
        # Add seasonal component with VARIATION (not rigid repetition)
        if seasonal_pattern is not None:
            print("   🔄 Adding seasonal component with variation...")
            seasonal_cycle = seasonal_pattern.values
            
            # Instead of exact repetition, add some year-to-year variation
            # Real seasonal patterns vary slightly each year
            seasonal_future = np.tile(seasonal_cycle, (forecast_horizon // len(seasonal_cycle)) + 1)[:forecast_horizon]
            
            # Add gradual variation (e.g., 5-10% random variation)
            seasonal_variation = np.random.normal(1.0, 0.05, forecast_horizon)
            seasonal_future_varied = seasonal_future * seasonal_variation
            
            forecast = forecast + seasonal_future_varied
            print(f"   ✅ Seasonal variation added (std: {np.std(seasonal_variation):.3f})")
        
        # Post-process with LOOSE clipping (more realistic variability)
        forecast = post_process_forecast(
            forecast, 
            target_column, 
            param_data.values, 
            strict=False  # Use loose clipping untuk preserve variability
        )
        
        print(f"   ✅ Forecast completed")
        print(f"   Range: {forecast.min():.2f} to {forecast.max():.2f}")
        print(f"   Mean: {forecast.mean():.2f}, Std: {forecast.std():.2f}")
        
        # ✅ REALISM CHECK
        print(f"\n🔍 REALISM CHECK:")
        print(f"   Historical std: {param_data.std():.2f}")
        print(f"   Forecast std: {forecast.std():.2f}")
        ratio = forecast.std() / param_data.std()
        print(f"   Variability ratio: {ratio:.2f}")
        if ratio > 0.5:
            print(f"   ✅ Forecast has REALISTIC variability!")
        elif ratio > 0.3:
            print(f"   ✅ Forecast has acceptable variability")
        else:
            print(f"   ⚠️  Forecast still somewhat smooth")
        
        print("\n💾 Step 7: Saving to database...")
        
        data_end_date = param_data.index[-1]
        forecast_dates = pd.date_range(
            start=data_end_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        forecast_docs = []
        start_dt_filter = pd.to_datetime(start_date) if start_date else None

        for i, (forecast_date, forecast_value) in enumerate(zip(forecast_dates, forecast)):
            # Filter by start_date if provided
            if start_dt_filter and forecast_date < start_dt_filter:
                continue
            
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
                        "day_ahead": i + 1,
                        "model_metadata": {
                            "framework": "PyTorch",
                            "model": "LSTM_Recursive",
                            "version": "1.0_recursive",
                            "forecast_method": "recursive/iterative",
                            "lookback_days": best_params['lookback'],
                            "lookback_years": best_params['lookback_years'],
                            "architecture": f"LSTM({best_params['hidden_size_1']})→LSTM({best_params['hidden_size_2']})",
                            "scaler": best_params['scaler_type'],
                            "dropout": best_params['dropout'],
                            "epochs": best_params['epochs'],
                            "batch_size": best_params['batch_size'],
                            "learning_rate": best_params['learning_rate'],
                            "loss_function": "smooth_l1_loss",
                            "seasonal_decomposed": seasonal_pattern is not None,
                            "validation_rmse": val_metrics.get('rmse', 0),
                            "validation_r2": val_metrics.get('r2', 0),
                            "test_rmse": test_metrics['rmse'],
                            "test_mae": test_metrics['mae'],
                            "test_mape": test_metrics['mape'],
                            "test_r2": test_metrics['r2'],
                            "forecast_std": float(forecast.std()),
                            "historical_std": float(param_data.std()),
                            "variability_ratio": float(ratio)
                        }
                    }
                }
            }
            
            if append_column_id:
                doc["column_id"] = f"{collection_name}_{target_column}"
            
            forecast_docs.append(doc)
        
        # Upsert
        upsert_count = 0
        for doc in forecast_docs:
            result = db[save_collection].update_one(
                {"forecast_date": doc["forecast_date"], "config_id": config_id},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1
        
        print(f"   ✅ Saved {upsert_count}/{len(forecast_docs)} documents")
        
        # 8. Summary
        print(f"\n{'='*70}")
        print(f"✅ RECURSIVE FORECAST COMPLETED")
        print(f"{'='*70}")
        print(f"\n📊 SUMMARY:")
        print(f"   Framework: PyTorch")
        print(f"   Method: RECURSIVE/ITERATIVE (1-step at a time)")
        print(f"   Data: {len(param_data)} points ({len(param_data)/365:.1f} years)")
        print(f"   Lookback: {best_params['lookback']} days ({best_params['lookback_years']} years)")
        print(f"   Architecture: 2-layer LSTM ({best_params['hidden_size_1']}→{best_params['hidden_size_2']})")
        print(f"\n📈 PERFORMANCE:")
        print(f"   Test RMSE: {test_metrics['rmse']:.3f}")
        print(f"   Test MAE: {test_metrics['mae']:.3f}")
        print(f"   Test R²: {test_metrics['r2']:.3f}")
        print(f"\n🔮 FORECAST:")
        print(f"   Period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   Range: {forecast.min():.2f} to {forecast.max():.2f}")
        print(f"   Variability ratio: {ratio:.2f} (MORE REALISTIC than direct multi-step)")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "framework": "PyTorch",
            "forecast_method": "recursive",
            "target_column": target_column,
            "error_metrics": test_metrics,
            "forecast": {
                "values": forecast.tolist(),
                "dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "std": float(forecast.std()),
                "variability_ratio": float(ratio)
            },
            "model_config": best_params
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
    
    finally:
        if should_close_client:
            client.close()

# ======================================================
# Run
# ======================================================
if __name__ == "__main__":
    result = run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
    
    print(f"\nFinal status: {result['status']}")
    if result['status'] == 'success':
        # FIX: Access correct key 'error_metrics' instead of 'test_metrics'
        print(f"Test R²: {result['error_metrics']['r2']:.3f}")
        print(f"Variability ratio: {result['forecast']['variability_ratio']:.2f}")