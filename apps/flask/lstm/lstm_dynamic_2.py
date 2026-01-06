import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Device configuration (Global fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
use_cuda = device.type == 'cuda'

RAINFALL_PARAMS = ["RR", "RR_imputed", "PRECTOTCORR"]
NDVI_PARAMS = ["NDVI", "NDVI_imputed"]
TEMP_PARAMS = ["TAVG", "TMAX", "TMIN", "T2M"]
HUMIDITY_PARAMS = ["RH_AVG", "RH_AVG_preprocessed", 'RH2M']
SOLAR_PARAMS = ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]

# ============================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================

class QuantileLoss(nn.Module):
    """
    Loss function untuk menangkap nilai ekstrem (puncak hujan).
    """
    def __init__(self, quantile=0.90):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        errors = target - preds
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.abs(loss).mean()

# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2, forecast_horizon=1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_sequences(data, seq_length, forecast_horizon=1):
    xs, ys = [], []
    # Safety check
    if len(data) <= seq_length + forecast_horizon:
        return np.array([]), np.array([])
        
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(xs), np.array(ys)

def convert_to_python_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    return obj

def is_rainfall(param_name):
    return param_name in RAINFALL_PARAMS

def is_ndvi(param_name):
    return param_name in NDVI_PARAMS

# ============================================================
# TRANSFORMATION & DECOMPOSITION
# ============================================================

def apply_log_transform(data, param_name):
    if not is_rainfall(param_name):
        return data, None
    data_array = np.array(data).flatten()
    transformed = np.log1p(data_array)
    return transformed, {'method': 'log1p'}

def inverse_log_transform(data, transform_params, param_name):
    if not is_rainfall(param_name) or transform_params is None:
        return data
    data_array = np.array(data).flatten()
    data_clipped = np.clip(data_array, -1, 7.5) 
    return np.expm1(data_clipped)

def seasonal_decompose_data(data, param_name):
    # Determine Period
    if is_ndvi(param_name):
        period = 23
    else:
        period = 365
    
    if len(data) < period * 2:
        period = max(7, len(data) // 3)
    
    try:
        stl = STL(data, period=period, robust=True)
        result = stl.fit()
        return {
            'trend': result.trend.values,
            'seasonal': result.seasonal.values,
            'residual': result.resid.values,
            'period': period
        }
    except Exception as e:
        print(f"⚠️ Decomposition failed (using raw data): {e}")
        return None

def extrapolate_seasonal(seasonal_pattern, forecast_steps, period, add_noise=True):
    """
    Extrapolate seasonal pattern with optional year-to-year variation
    
    Args:
        add_noise: If True, add realistic seasonal shifts (5-8% variation)
    """
    one_cycle = seasonal_pattern[-period:]
    n_repeats = (forecast_steps // period) + 2
    extended = np.tile(one_cycle, n_repeats)[:forecast_steps]
    
    if add_noise:
        # Add realistic year-to-year seasonal variation (El Niño, climate shifts)
        seasonal_std = np.std(one_cycle)
        noise_level = 0.07  # 7% variation (conservative)
        
        # Generate smooth seasonal noise (not random spikes)
        # Use longer period noise to simulate gradual seasonal shifts
        smooth_noise = np.zeros(forecast_steps)
        for i in range(0, forecast_steps, period // 4):  # 4 shifts per year
            smooth_noise[i:i + period // 4] = np.random.normal(0, seasonal_std * noise_level)
        
        # Smooth the noise (simple moving average if scipy unavailable)
        try:
            from scipy.ndimage import gaussian_filter1d
            smooth_noise = gaussian_filter1d(smooth_noise, sigma=20)
        except ImportError:
            # Fallback: simple moving average
            window = 40
            kernel = np.ones(window) / window
            smooth_noise = np.convolve(smooth_noise, kernel, mode='same')
        
        extended = extended + smooth_noise[:forecast_steps]
    
    return extended

def extrapolate_trend(trend, forecast_steps):
    window = min(30, len(trend) // 4)
    last_values = trend[-window:]
    slope = (last_values[-1] - last_values[0]) / len(last_values) if len(last_values) > 1 else 0
    
    forecast_trend = []
    current_value = trend[-1]
    for _ in range(forecast_steps):
        current_value += slope
        forecast_trend.append(current_value)
    return np.array(forecast_trend)

# ============================================================
# POST PROCESSING
# ============================================================

def post_process_forecast(forecast, param_name, historical_data=None, transform_params=None):
    forecast = np.array(forecast).flatten()
    
    if is_rainfall(param_name):
        if transform_params is not None:
            forecast = inverse_log_transform(forecast, transform_params, param_name)
        
        forecast = np.where(forecast < 0.5, 0, forecast)
        
        if historical_data is not None:
            hist_max = float(np.max(historical_data))
            hist_p99 = float(np.percentile(historical_data, 99))
            # Allow extreme events up to 3x historical max or 4x P99
            max_cap = min(hist_max * 3.0, hist_p99 * 4.0, 450)
            forecast = np.clip(forecast, 0, max_cap)
        else:
            forecast = np.clip(forecast, 0, 400)
            
        forecast = np.round(forecast, 1)

    elif is_ndvi(param_name):
        forecast = np.clip(forecast, -1, 1)
    elif param_name in HUMIDITY_PARAMS:
        forecast = np.clip(forecast, 0, 100)
    elif param_name in TEMP_PARAMS:
        forecast = np.clip(forecast, 15, 45)
    elif param_name in SOLAR_PARAMS:
        forecast = np.clip(forecast, 0, 40)
    
    return forecast

def check_variability(forecast, historical_data, param_name):
    """Check if forecast has realistic variability (not too smooth)
    
    Uses METEOROLOGICAL STANDARDS for thresholds:
    - TAVG: 0.3°C (WMO standard for daily temperature significance)
    - RH_AVG: 1.0% (typical humidity measurement precision)
    - RR: 0.5mm (rainfall measurement precision)
    - SOLAR: 0.5 MJ/m² (cloud cover sensitivity standard)
    """
    hist_std = float(historical_data.std())
    forecast_std = float(np.std(forecast))
    std_ratio = (forecast_std / hist_std * 100) if hist_std > 0 else 0
    
    # Calculate flat days percentage with METEOROLOGICAL STANDARDS
    daily_changes = np.abs(np.diff(forecast))
    if is_rainfall(param_name):
        threshold = 0.5  # mm (measurement precision)
        target_flat = 20  # target <20% flat days
    elif param_name in TEMP_PARAMS:
        threshold = 0.3  # °C (WMO standard for significance)
        target_flat = 40  # target <40% flat days
    elif param_name in SOLAR_PARAMS:
        threshold = 0.5  # MJ/m² (cloud cover sensitivity)
        target_flat = 40  # target <40% flat days
    else:
        threshold = 1.0  # % (humidity default)
        target_flat = 40
    
    flat_days = np.sum(daily_changes < threshold)
    flat_percentage = (flat_days / len(daily_changes)) * 100 if len(daily_changes) > 0 else 0
    
    print(f"\n📊 Variability Check:")
    print(f"   Std Ratio: {std_ratio:.1f}% | Flat Days: {flat_percentage:.1f}% (target <{target_flat}%)")
    
    if flat_percentage < target_flat:
        print(f"   ✅ Quality OK")
        return True
    else:
        print(f"   ⚠️ TOO SMOOTH ({flat_percentage:.1f}% > {target_flat}%)")
        return False

# ============================================================
# TRAINING LOGIC
# ============================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, hidden_size, num_layers, 
                       dropout, lr, batch_size, max_epochs, patience,
                       train_scaled, seq_len, scaler, decomposition,
                       split_point, train_data, param_name, transform_params=None,
                       forecast_horizon=1):
    global use_cuda
    current_device = torch.device("cuda" if use_cuda else "cpu")
    
    model = LSTMModel(1, hidden_size, num_layers, dropout, forecast_horizon).to(current_device)
    
    # SMART LOSS SELECTION
    if is_rainfall(param_name):
        criterion = QuantileLoss(quantile=0.85)
    elif param_name in TEMP_PARAMS or param_name in HUMIDITY_PARAMS:
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.from_numpy(X_train).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(y_train).float()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for _ in range(max_epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(current_device), batch_y.to(current_device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1).to(current_device)
            y_val_t = torch.from_numpy(y_val).float().to(current_device)
            val_loss = criterion(model(X_val_t), y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    # Evaluate on Validation Set
    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1).to(current_device)
        forecast_scaled = model(X_val_t).cpu().numpy()[0]
    
    forecast_residual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    
    # Reconstruct logic for evaluation
    if decomposition is not None:
        forecast_len = len(forecast_residual)
        # Use simple slice for evaluation reconstruction
        val_seasonal = decomposition['seasonal'][split_point:split_point + forecast_len]
        val_trend = decomposition['trend'][split_point:split_point + forecast_len]
        
        # Handle lengths
        min_len = min(len(forecast_residual), len(val_seasonal), len(val_trend))
        forecast = forecast_residual[:min_len] + val_seasonal[:min_len] + val_trend[:min_len]
    else:
        forecast = forecast_residual
    
    # Post process & Metric Calc
    forecast = post_process_forecast(forecast, param_name, train_data.values, transform_params)
    actual = train_data.iloc[split_point:split_point + len(forecast)].values
    min_len = min(len(actual), len(forecast))
    
    mse = float(mean_squared_error(actual[:min_len], forecast[:min_len]))
    mae = float(mean_absolute_error(actual[:min_len], forecast[:min_len]))
    
    # Calculate actual epochs trained (max_epochs - early_stop)
    actual_epochs = max_epochs - max(0, patience - epochs_no_improve)
    
    metrics = {'mae': mae, 'mse': mse}
    return metrics, actual_epochs

def grid_search_lstm_params(train_data, param_name, validation_ratio=0.10):
    print(f"\n{'='*60}\n🔍 SMART GRID SEARCH: {param_name}\n{'='*60}")
    
    # 1. Prepare Data
    if is_rainfall(param_name):
        transformed_data, transform_params = apply_log_transform(train_data.values, param_name)
        working_series = pd.Series(transformed_data, index=train_data.index)
    else:
        transform_params = None
        working_series = train_data
    
    decomposition = seasonal_decompose_data(working_series, param_name)
    if decomposition is not None:
        working_data = pd.Series(decomposition['residual'], index=working_series.index).ffill().bfill()
    else:
        working_data = working_series
        
    val_size = max(30, int(len(working_data) * validation_ratio))
    split_point = len(working_data) - val_size
    train_split = working_data[:split_point]
    val_split = working_data[split_point:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_split.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_split.values.reshape(-1, 1)).flatten()
    
    # 2. CONFIGURATION
    if is_rainfall(param_name):
        config = {
            'hidden_sizes': [64, 128], 'seq_lengths': [30, 90],
            'learning_rates': [0.001], 'batch_sizes': [16],
            'dropout': 0.25, 'scout_epochs': 25
        }
    else:
        config = {
            'hidden_sizes': [32, 64], 'seq_lengths': [14, 30],
            'learning_rates': [0.002], 'batch_sizes': [32],
            'dropout': 0.15, 'scout_epochs': 20
        }
    
    best_aic = float('inf')
    best_params = None
    best_metrics = None
    val_horizon = min(len(val_split), 30)
    
    # 3. SEARCH LOOP
    for seq_len in config['seq_lengths']:
        X_train, y_train = create_sequences(train_scaled, seq_len, val_horizon)
        X_val, y_val = create_sequences(val_scaled, seq_len, val_horizon)
        
        if len(X_train) == 0: continue
        
        for hidden in config['hidden_sizes']:
            for lr in config['learning_rates']:
                print(f"   ⚡ Testing Seq={seq_len}, Hid={hidden}...", end=" ")
                
                result = train_and_evaluate(
                    X_train, y_train, X_val, y_val, 
                    hidden, 2, config['dropout'], lr, config['batch_sizes'][0],
                    config['scout_epochs'], 8,
                    train_scaled, seq_len, scaler, decomposition, split_point, train_data, param_name, transform_params,
                    val_horizon
                )
                
                metrics, _ = result
                score = metrics['mse']
                print(f"MAE={metrics['mae']:.3f} | Score={score:.5f}")
                
                if score < best_aic:
                    best_aic = score
                    best_params = {
                        'hidden_size': hidden, 'num_layers': 2, 'learning_rate': lr,
                        'seq_length': seq_len, 'dropout': config['dropout'], 
                        'batch_size': config['batch_sizes'][0],
                        'epochs': 70 if is_rainfall(param_name) else 50 
                    }
                    best_metrics = metrics
                    print(f"      ✅ Candidate Found!")
    
    # SAFETY FALLBACK if grid search failed
    if best_params is None:
        print("   ⚠️ Grid search failed (insufficient data). Using defaults.")
        best_params = {
            'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.001,
            'seq_length': min(14, len(train_split)//2), 'dropout': 0.2, 
            'batch_size': 16, 'epochs': 50
        }
        
    return best_params, best_metrics, decomposition, transform_params

def fit_lstm_model(data, best_params, param_name, forecast_steps):
    global use_cuda
    final_epochs = best_params['epochs']
    print(f"\n🧪 Fitting Final Model (Quality Mode) | Steps: {forecast_steps} | Epochs: {final_epochs}")
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(data_scaled, best_params['seq_length'], forecast_steps)
    
    # Handling insufficient data for forecast horizon
    if len(X) == 0:
        forecast_steps = max(1, len(data) - best_params['seq_length']) # Ensure at least 1 step
        X, y = create_sequences(data_scaled, best_params['seq_length'], forecast_steps)
        print(f"   ⚠️ Adjusted forecast steps to {forecast_steps}")

    current_device = torch.device("cuda" if use_cuda else "cpu")
    model = LSTMModel(1, best_params['hidden_size'], 2, best_params['dropout'], forecast_steps).to(current_device)
    
    if is_rainfall(param_name):
        criterion = QuantileLoss(quantile=0.85)
    elif param_name in TEMP_PARAMS:
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(y).float()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    for epoch in range(final_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(current_device), batch_y.to(current_device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{final_epochs} | Loss: {epoch_loss/len(loader):.6f}")
            
    return model, scaler, forecast_steps

# ============================================================
# MAIN
# ============================================================

def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast", 
                      config_id=None, append_column_id=True, client=None, start_date=None, end_date=None):
    
    # Use global device configuration
    global device
    print(f"\n{'#'*60}\n# LSTM ANALYSIS: {target_column}\n{'#'*60}")
    
    if client is None:
        load_dotenv()
        client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["tugas_akhir"]
    source_data = list(db[collection_name].find().sort("Date", 1))
    
    # Data Prep
    df = pd.DataFrame(source_data)
    date_col = next((c for c in ['Date', 'date', 'timestamp'] if c in df.columns), None)
    df['timestamp'] = pd.to_datetime(df[date_col])
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    # Fix: bfill after interpolate to handle starting NaNs
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='D')).interpolate(method='linear').bfill()
    param_data = df[target_column].dropna()
    
    # Grid Search
    best_params, error_metrics, decomposition, transform_params = grid_search_lstm_params(param_data, target_column)
    
    # Forecast Setup
    last_hist_date = param_data.index[-1]
    if start_date:
        forecast_start = pd.to_datetime(start_date)
        forecast_end = pd.to_datetime(end_date)
    else:
        forecast_start = last_hist_date + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=364)
    
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
    forecast_steps = len(forecast_dates)
    
    # Prepare Data for Final Model
    if decomposition:
        train_data_final = pd.Series(decomposition['residual'], index=param_data.index).ffill().bfill()
    else:
        train_data_final = param_data
        
    # Fit Final Model
    model, scaler, actual_steps = fit_lstm_model(train_data_final, best_params, target_column, forecast_steps)
    if actual_steps != forecast_steps:
        forecast_dates = forecast_dates[:actual_steps]
        forecast_steps = actual_steps
        
    # Inference
    data_scaled = scaler.transform(train_data_final.values.reshape(-1, 1)).flatten()
    input_seq = data_scaled[-best_params['seq_length']:].copy()
    
    # SINGLE INFERENCE (No MC Dropout - use residual bootstrapping instead)
    print(f"\n🎯 LSTM Inference (single pass)...")
    model.eval()
    
    with torch.no_grad():
        seq_t = torch.from_numpy(input_seq).float().unsqueeze(0).unsqueeze(-1).to(device)
        forecast_scaled = model(seq_t).cpu().numpy()[0]
    
    forecast_residual = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    print(f"   ✅ Inference completed.")
    
    # 🧬 RECONSTRUCTION WITH RESIDUAL BOOTSTRAPPING
    if decomposition:
        print(f"\n🧬 Reconstructing with Residual Bootstrapping...")
        forecast_seasonal = extrapolate_seasonal(
            decomposition['seasonal'], 
            forecast_steps, 
            decomposition['period'],
            add_noise=True  # Add year-to-year seasonal variation
        )
        print(f"   🌍 Seasonal variation injected (7% natural shifts)")
        forecast_trend = extrapolate_trend(decomposition['trend'], forecast_steps)
        
        # RESIDUAL BOOTSTRAPPING - Sample from historical residuals
        hist_residuals = decomposition['residual']
        hist_resid_std = np.std(hist_residuals)
        
        if is_rainfall(target_column):
            # Rainfall: Bootstrap with gamma distribution for asymmetry
            sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
            gamma_shape = np.random.gamma(2.0, 1.0, forecast_steps)
            gamma_shape = (gamma_shape - np.mean(gamma_shape)) / (np.std(gamma_shape) + 1e-6)
            noise = sampled_residuals * 0.95 + hist_resid_std * 0.20 * gamma_shape
            forecast_residual = forecast_residual + noise
            print(f"   📊 Applied Residual Bootstrap (95% sample + 20% gamma, Rainfall)")
            
        else:
            # Other parameters: Direct bootstrapping with multipliers
            if target_column in TEMP_PARAMS:
                # Temperature: 110% strength
                sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                noise = sampled_residuals * 1.10
                forecast_residual = forecast_residual + noise
                print(f"   📊 Applied Residual Bootstrap (110% strength - Temp)")
                
            elif target_column in HUMIDITY_PARAMS:
                # CORRELATION-AWARE RESIDUAL BOOTSTRAPPING
                try:
                    # Try to load TAVG forecast for correlation
                    tavg_forecast = list(db[save_collection].find(
                        {"config_id": config_id}, 
                        sort=[("forecast_date", 1)]
                    ).limit(forecast_steps))
                    
                    tavg_values = None
                    if tavg_forecast and 'TAVG' in tavg_forecast[0].get('parameters', {}):
                        tavg_values = np.array([
                            doc['parameters']['TAVG']['forecast_value'] 
                            for doc in tavg_forecast
                        ])
                        print(f"   🔗 Loading TAVG for correlation...")
                    else:
                        # Generate TAVG reference from historical
                        tavg_col = next((c for c in df.columns if c in TEMP_PARAMS), None)
                        if tavg_col:
                            tavg_hist = df[tavg_col].dropna()
                            tavg_decomp = seasonal_decompose_data(tavg_hist, tavg_col)
                            if tavg_decomp:
                                tavg_seasonal_fc = extrapolate_seasonal(
                                    tavg_decomp['seasonal'], 
                                    forecast_steps, 
                                    tavg_decomp['period'],
                                    add_noise=True
                                )
                                tavg_trend_fc = extrapolate_trend(tavg_decomp['trend'], forecast_steps)
                                # Use TAVG residuals for correlation pattern
                                tavg_residuals = tavg_decomp['residual']
                                tavg_sampled = np.random.choice(tavg_residuals, size=forecast_steps, replace=True)
                                tavg_values = tavg_seasonal_fc + tavg_trend_fc + tavg_sampled * 0.85
                                print(f"   📊 Generated TAVG reference (85% residuals)...")
                    
                    if tavg_values is not None and len(tavg_values) == len(forecast_residual):
                        # Correlated residual bootstrapping
                        sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                        base_noise = sampled_residuals * 0.65
                        
                        # Add anti-correlation with TAVG (UNCHANGED - correlation preserved!)
                        tavg_normalized = (tavg_values - np.mean(tavg_values)) / (np.std(tavg_values) + 1e-6)
                        correlation_component = -0.45 * tavg_normalized * hist_resid_std
                        
                        noise = base_noise + correlation_component
                        print(f"   📊 Applied Correlated Bootstrap (65% + anti-corr 45%)")
                    else:
                        sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                        noise = sampled_residuals * 0.95
                        print(f"   📊 Applied Residual Bootstrap (95% - independent)")
                        
                except Exception as e:
                    sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                    noise = sampled_residuals * 0.95
                    print(f"   📊 Applied Residual Bootstrap (95% - fallback)")
                    
                forecast_residual = forecast_residual + noise
                    
            elif target_column in SOLAR_PARAMS:
                # Solar: 105% strength
                sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                noise = sampled_residuals * 1.05
                forecast_residual = forecast_residual + noise
                print(f"   📊 Applied Residual Bootstrap (105% strength - Solar)")
            else:
                # Default: 100% strength
                sampled_residuals = np.random.choice(hist_residuals, size=forecast_steps, replace=True)
                noise = sampled_residuals * 1.0
                forecast_residual = forecast_residual + noise
                print(f"   📊 Applied Residual Bootstrap (100% - default)")
            
        forecast = forecast_residual + forecast_seasonal + forecast_trend
    else:
        # No decomposition: Direct residual bootstrapping not possible, use gaussian
        hist_std = np.std(train_data_final.values)
        if is_rainfall(target_column):
            noise = hist_std * 1.2 * np.random.gamma(2.0, 1.0, len(forecast_residual))
        else:
            noise = hist_std * 1.0 * np.random.randn(len(forecast_residual))
        forecast = forecast_residual + noise
        print(f"   ⚠️ No decomposition: using gaussian noise (fallback)")

    # Post-process and check quality
    forecast = post_process_forecast(forecast, target_column, param_data.values, transform_params)
    check_variability(forecast, param_data, target_column)
    
    print(f"\n💾 Saving {len(forecast)} days to MongoDB...")
    upsert_count = 0
    for i, date in enumerate(forecast_dates):
        doc = {
            "forecast_date": date.to_pydatetime(),
            "timestamp": datetime.now().isoformat(),
            "config_id": config_id,
            "parameters": {
                target_column: {
                    "forecast_value": float(forecast[i]),
                    "model_metadata": best_params
                }
            }
        }
        if append_column_id: doc["column_id"] = f"{collection_name}_{target_column}"
            
        db[save_collection].update_one(
            {"forecast_date": doc["forecast_date"], "config_id": config_id},
            {"$set": {f"parameters.{target_column}": doc["parameters"][target_column]}},
            upsert=True
        )
        upsert_count += 1
        
    print(f"✅ Completed. Saved {upsert_count} docs.")
    return {"status": "success", "count": upsert_count}

if __name__ == "__main__":
    # Example usage
    run_lstm_analysis(collection_name="bmkg-data", target_column="RR")