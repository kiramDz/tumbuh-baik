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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
use_cuda = device.type == 'cuda'

RAINFALL_PARAMS = ["RR", "RR_imputed", "PRECTOTCORR"]
NDVI_PARAMS = ["NDVI", "NDVI_imputed"]
TEMP_PARAMS = ["TAVG", "TMAX", "TMIN", "T2M"]
HUMIDITY_PARAMS = ["RH_AVG", "RH_AVG_preprocessed", 'RH2M']
SOLAR_PARAMS = ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"]


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
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


def calculate_aic(mse, n_params, n_samples):
    if mse <= 0 or n_samples <= 0:
        return float('inf')
    return 2 * n_params + n_samples * np.log(mse)


def is_rainfall(param_name):
    return param_name in RAINFALL_PARAMS


def is_ndvi(param_name):
    return param_name in NDVI_PARAMS


# ============================================================
# TRANSFORMATION FUNCTIONS (untuk rainfall)
# ============================================================

def apply_log_transform(data, param_name):
    """Apply log1p transform untuk rainfall"""
    if not is_rainfall(param_name):
        return data, None
    
    data_array = np.array(data).flatten()
    transformed = np.log1p(data_array)
    
    print(f"\n📦 Log1p Transform:")
    print(f"   Original: [{data_array.min():.2f}, {data_array.max():.2f}], std={data_array.std():.3f}")
    print(f"   Transformed: [{transformed.min():.2f}, {transformed.max():.2f}], std={transformed.std():.3f}")
    
    return transformed, {'method': 'log1p'}


def inverse_log_transform(data, transform_params, param_name):
    """Inverse log1p transform"""
    if not is_rainfall(param_name) or transform_params is None:
        return data
    
    data_array = np.array(data).flatten()
    # Clip untuk mencegah overflow - KONSISTEN dengan post_process_forecast
    data_clipped = np.clip(data_array, -10, 6.5)
    return np.expm1(data_clipped)


# ============================================================
# DECOMPOSITION FUNCTIONS
# ============================================================

def get_period(param_name, data_length):
    if is_ndvi(param_name):
        period = 23
    else:
        period = 365
    
    if data_length < period * 2:
        period = max(7, data_length // 3)
        print(f"   ⚠️ Adjusted period to {period}")
    
    return period


def seasonal_decompose_data(data, param_name):
    """Decompose time series"""
    period = get_period(param_name, len(data))
    
    try:
        stl = STL(data, period=period, robust=True)
        result = stl.fit()
        
        decomposition = {
            'trend': result.trend.values,
            'seasonal': result.seasonal.values,
            'residual': result.resid.values,
            'period': period
        }
        
        print(f"✓ Decomposition (period={period}):")
        print(f"   Trend: [{result.trend.min():.3f}, {result.trend.max():.3f}]")
        print(f"   Seasonal: [{result.seasonal.min():.3f}, {result.seasonal.max():.3f}]")
        print(f"   Residual std: {result.resid.std():.4f}")
        
        return decomposition
    except Exception as e:
        print(f"❌ Decomposition failed: {e}")
        return None


def extrapolate_seasonal(seasonal_pattern, forecast_steps, period, param_name=None):
    """Extrapolate seasonal pattern dengan parameter-specific dampening"""
    one_cycle = seasonal_pattern[-period:]
    n_repeats = (forecast_steps // period) + 2
    extended = np.tile(one_cycle, n_repeats)[:forecast_steps]
    
    if param_name is not None and is_rainfall(param_name):
        seasonal_std = np.std(one_cycle)
        seasonal_mean = np.mean(one_cycle)
        
        # PERBAIKAN: Lebih longgar untuk preserve high rainfall events
        lower_bound = seasonal_mean - 4.0 * seasonal_std  # Was 3.0
        upper_bound = seasonal_mean + 4.0 * seasonal_std  # Was 3.0
        extended = np.clip(extended, lower_bound, upper_bound)
        
        # Cap maksimum lebih tinggi untuk allow extreme events
        extended = np.clip(extended, -2.0, 6.0)  # Was 5.5, log1p(400) ≈ 6.0
        
        print(f"   📉 Seasonal dampened: [{extended.min():.3f}, {extended.max():.3f}]")
    
    return extended


def extrapolate_trend(trend, forecast_steps, param_name=None):
    """
    Extrapolate trend dengan parameter-specific dampening dan constraints
    """
    window = min(30, len(trend) // 4)
    last_values = trend[-window:]
    slope = (last_values[-1] - last_values[0]) / len(last_values) if len(last_values) > 1 else 0
    
    forecast_trend = []
    current_value = trend[-1]
    current_slope = slope
    
    # Parameter-specific dampening dan constraints
    if param_name in TEMP_PARAMS:
        damping = 0.85  # Sangat agresif - suhu stabil
        trend_mean = np.mean(trend[-365:]) if len(trend) > 365 else np.mean(trend)
        max_deviation = 1.5  # Max ±1.5°C dari trend mean
        
    elif param_name in HUMIDITY_PARAMS:
        damping = 0.88  # Agresif - kelembaban relatif stabil
        trend_mean = np.mean(trend[-365:]) if len(trend) > 365 else np.mean(trend)
        max_deviation = 5.0  # Max ±5% dari trend mean
        
    elif is_rainfall(param_name):
        damping = 0.95  # Moderate - rainfall boleh variabel tapi tidak ekstrem
        trend_mean = np.mean(trend[-365:]) if len(trend) > 365 else np.mean(trend)
        max_deviation = trend_mean * 0.5  # Max ±50% dari trend mean
        
    elif param_name in SOLAR_PARAMS:
        damping = 0.90  # Moderate-agresif - radiasi cukup stabil
        trend_mean = np.mean(trend[-365:]) if len(trend) > 365 else np.mean(trend)
        max_deviation = 3.0  # Max ±3 MJ/m² dari trend mean
        
    else:
        damping = 0.93
        trend_mean = None
        max_deviation = None
    
    for step in range(forecast_steps):
        current_value += current_slope
        current_slope *= damping
        
        # Apply mean-reversion constraint
        if trend_mean is not None and max_deviation is not None:
            deviation = abs(current_value - trend_mean)
            if deviation > max_deviation:
                # Strong pull back to mean
                pull_strength = min(0.3, (deviation - max_deviation) / max_deviation)
                current_value = current_value - pull_strength * (current_value - trend_mean)
                current_slope *= 0.5  # Reduce slope aggressively
        
        forecast_trend.append(current_value)
    
    return np.array(forecast_trend)


# ============================================================
# POST PROCESSING
# ============================================================

def post_process_forecast(forecast, param_name, historical_data=None, transform_params=None):
    """Post-process forecast values dengan parameter-specific logic"""
    forecast = np.array(forecast).flatten()
    
    if is_rainfall(param_name):
        if transform_params is not None:
            forecast = np.clip(forecast, -5, 6.5)
            forecast = inverse_log_transform(forecast, transform_params, param_name)
            print(f"   ✓ Log1p inverse applied")
        
        forecast = np.where(forecast < 0.5, 0, forecast)
        
        if historical_data is not None:
            hist_max = float(np.max(historical_data))
            hist_p99 = float(np.percentile(historical_data, 99))
            hist_p95 = float(np.percentile(historical_data, 95))
            hist_mean = float(np.mean(historical_data))
            hist_std = float(np.std(historical_data))
            
            # PERBAIKAN: Allow high extremes untuk rainfall
            max_cap = max(
                hist_max * 1.5,           # 1.5x max historis
                hist_p99 * 2.0,           # 2x P99
                hist_p95 * 3.0,           # 3x P95
                hist_mean + 6 * hist_std  # Mean + 6 std
            )
            max_cap = min(max_cap, 150)  # Hard cap 150mm (was 500)
            
            print(f"   📊 RAIN Cap: max={hist_max:.1f}, P99={hist_p99:.1f}, cap={max_cap:.1f}")
        else:
            max_cap = 100
        
        forecast = np.clip(forecast, 0, max_cap)
        forecast = np.round(forecast, 1)
        
    elif is_ndvi(param_name):
        forecast = np.clip(forecast, -1, 1)
        
    elif param_name in HUMIDITY_PARAMS:
        # PERBAIKAN: Constraint berdasarkan historis
        if historical_data is not None:
            hist_mean = float(np.mean(historical_data))
            hist_std = float(np.std(historical_data))
            hist_min = float(np.min(historical_data))
            hist_max = float(np.max(historical_data))
            
            # Allow ±3 std dari mean, tapi tidak keluar dari hist range terlalu jauh
            lower_bound = max(hist_mean - 3 * hist_std, hist_min - 5.0, 50.0)
            upper_bound = min(hist_mean + 3 * hist_std, hist_max + 5.0, 95.0)
            
            forecast = np.clip(forecast, lower_bound, upper_bound)
            print(f"   📊 HUM Cap: mean={hist_mean:.1f}%, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        else:
            forecast = np.clip(forecast, 50, 95)
        
        forecast = np.round(forecast, 1)
        
    elif param_name in TEMP_PARAMS:
        # PERBAIKAN: Constraint ketat berdasarkan historis
        if historical_data is not None:
            hist_mean = float(np.mean(historical_data))
            hist_std = float(np.std(historical_data))
            hist_min = float(np.min(historical_data))
            hist_max = float(np.max(historical_data))
            
            lower_bound = max(hist_mean - 3 * hist_std, hist_min - 2.0, 20.0)
            upper_bound = min(hist_mean + 3 * hist_std, hist_max + 2.0, 36.0)
            
            forecast = np.clip(forecast, lower_bound, upper_bound)
            print(f"   📊 TEMP Cap: mean={hist_mean:.1f}°C, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        else:
            forecast = np.clip(forecast, 20, 36)
        
        forecast = np.round(forecast, 2)
        
    elif param_name in SOLAR_PARAMS:
        # PERBAIKAN: Constraint berdasarkan historis
        if historical_data is not None:
            hist_mean = float(np.mean(historical_data))
            hist_std = float(np.std(historical_data))
            hist_min = float(np.min(historical_data))
            hist_max = float(np.max(historical_data))
            
            lower_bound = max(hist_mean - 3 * hist_std, hist_min - 2.0, 8.0)
            upper_bound = min(hist_mean + 3 * hist_std, hist_max + 2.0, 28.0)
            
            forecast = np.clip(forecast, lower_bound, upper_bound)
            print(f"   📊 SOLAR Cap: mean={hist_mean:.1f} MJ/m², bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        else:
            forecast = np.clip(forecast, 8, 28)
        
        forecast = np.round(forecast, 2)
    
    return forecast


# ============================================================
# GAP FILLING FUNCTIONS
# ============================================================

def apply_parameter_constraints(value, param_name):
    """Apply realistic constraints based on parameter type"""
    if is_rainfall(param_name):
        return max(0, min(value, 200))  # 0-200mm
    elif param_name in HUMIDITY_PARAMS:
        return max(0, min(value, 100))  # 0-100%
    elif param_name in TEMP_PARAMS:
        return max(15, min(value, 45))  # 15-45°C
    elif param_name in SOLAR_PARAMS:
        return max(0, min(value, 35))   # 0-35 MJ/m²
    elif is_ndvi(param_name):
        return max(-1, min(value, 1))   # -1 to 1
    return value


def fill_gap_with_synthetic_data(param_data, last_historical_date, target_start_date, param_name, decomposition=None):
    """
    Fill gap between historical data and forecast start with synthetic data.
    Uses seasonal pattern matching from previous years for realistic values.
    
    Args:
        param_data: Historical data series
        last_historical_date: Last date with actual data
        target_start_date: User's desired forecast start date
        param_name: Parameter name for constraints
        decomposition: Optional decomposition dict for fallback
    
    Returns:
        extended_data: Historical + synthetic data
        gap_fill_info: Metadata about gap filling
    """
    gap_days = (target_start_date - last_historical_date).days - 1
    
    if gap_days <= 0:
        return param_data, None
    
    print(f"\n{'='*60}")
    print(f"🔧 GAP FILLING: {param_name}")
    print(f"{'='*60}")
    print(f"   Generating {gap_days} synthetic data points")
    print(f"   From: {(last_historical_date + pd.Timedelta(days=1)).date()}")
    print(f"   To: {(target_start_date - pd.Timedelta(days=1)).date()}")
    
    # Generate gap dates
    freq = '16D' if is_ndvi(param_name) else 'D'
    gap_start = last_historical_date + pd.Timedelta(days=1)
    gap_end = target_start_date - pd.Timedelta(days=1)
    gap_dates = pd.date_range(start=gap_start, end=gap_end, freq=freq)
    
    if len(gap_dates) == 0:
        print(f"   ℹ️ No gap dates to fill")
        return param_data, None
    
    # Calculate historical statistics for noise generation
    hist_std = param_data.std()
    hist_mean = param_data.mean()
    
    # METHOD: Seasonal Pattern Matching from previous years
    synthetic_values = []
    matched_count = 0
    fallback_count = 0
    
    for gap_date in gap_dates:
        historical_values = []
        
        # Look for same date in previous 5 years
        for year_offset in range(1, 6):
            lookup_date = gap_date - pd.DateOffset(years=year_offset)
            
            # Try exact date first
            if lookup_date in param_data.index:
                historical_values.append(param_data.loc[lookup_date])
            else:
                # Try nearby dates (±3 days) if exact not found
                for day_offset in range(-3, 4):
                    nearby_date = lookup_date + pd.Timedelta(days=day_offset)
                    if nearby_date in param_data.index:
                        historical_values.append(param_data.loc[nearby_date])
                        break
        
        if historical_values:
            # Use mean of previous years + controlled noise
            base_value = np.mean(historical_values)
            std_value = np.std(historical_values) if len(historical_values) > 1 else hist_std * 0.1
            
            # Add small noise (10% of local std)
            noise = np.random.normal(0, std_value * 0.1)
            synthetic_value = base_value + noise
            matched_count += 1
        else:
            # Fallback: Use decomposition if available
            if decomposition is not None:
                period = decomposition['period']
                day_of_year = gap_date.dayofyear
                seasonal_idx = day_of_year % len(decomposition['seasonal'])
                
                seasonal_value = decomposition['seasonal'][seasonal_idx]
                trend_value = decomposition['trend'][-1]
                
                # Add small noise
                noise = np.random.normal(0, hist_std * 0.05)
                synthetic_value = trend_value + seasonal_value + noise
            else:
                # Last resort: Use recent average with decay
                recent_values = param_data.tail(30).values
                synthetic_value = np.mean(recent_values) + np.random.normal(0, hist_std * 0.1)
            
            fallback_count += 1
        
        # Apply parameter-specific constraints
        synthetic_value = apply_parameter_constraints(synthetic_value, param_name)
        synthetic_values.append(synthetic_value)
    
    # Create synthetic series
    synthetic_series = pd.Series(synthetic_values, index=gap_dates)
    
    # Combine historical + synthetic
    extended_data = pd.concat([param_data, synthetic_series]).sort_index()
    
    # Remove duplicates if any
    extended_data = extended_data[~extended_data.index.duplicated(keep='first')]
    
    # Statistics
    print(f"\n   📊 Gap Fill Statistics:")
    print(f"   Pattern matched: {matched_count}/{len(gap_dates)} ({matched_count/len(gap_dates)*100:.1f}%)")
    print(f"   Fallback used: {fallback_count}/{len(gap_dates)} ({fallback_count/len(gap_dates)*100:.1f}%)")
    print(f"   Synthetic range: [{min(synthetic_values):.2f}, {max(synthetic_values):.2f}]")
    print(f"   Synthetic mean: {np.mean(synthetic_values):.2f} (hist: {hist_mean:.2f})")
    print(f"   Synthetic std: {np.std(synthetic_values):.2f} (hist: {hist_std:.2f})")
    print(f"   Extended data: {len(param_data)} → {len(extended_data)} records")
    print(f"{'='*60}\n")
    
    gap_fill_info = {
        "method": "seasonal_pattern_matching",
        "gap_days_filled": len(gap_dates),
        "gap_start": gap_start.strftime("%Y-%m-%d"),
        "gap_end": gap_end.strftime("%Y-%m-%d"),
        "pattern_matched_pct": round(matched_count / len(gap_dates) * 100, 1),
        "fallback_used_pct": round(fallback_count / len(gap_dates) * 100, 1),
        "synthetic_mean": float(np.mean(synthetic_values)),
        "synthetic_std": float(np.std(synthetic_values)),
        "historical_mean": float(hist_mean),
        "historical_std": float(hist_std)
    }
    
    return extended_data, gap_fill_info


# ============================================================
# VARIABILITY CHECK
# ============================================================

def check_variability(forecast, historical_data, param_name):
    hist_std = float(historical_data.std())
    forecast_std = float(np.std(forecast))
    std_ratio = (forecast_std / hist_std * 100) if hist_std > 0 else 0
    
    print(f"\n📊 Variability Check ({param_name}):")
    print(f"   Historical: mean={historical_data.mean():.3f}, std={hist_std:.3f}")
    print(f"   Forecast: mean={np.mean(forecast):.3f}, std={forecast_std:.3f}")
    print(f"   Std Ratio: {std_ratio:.1f}%")
    
    q_size = len(forecast) // 4
    for i in range(4):
        q = forecast[i * q_size:(i + 1) * q_size]
        print(f"   Q{i+1}: mean={np.mean(q):.3f}, range=[{q.min():.2f}, {q.max():.2f}]")
    
    if 40 <= std_ratio <= 160:
        print(f"   ✅ Variability OK")
        return True
    
    status = "LOW" if std_ratio < 40 else "HIGH"
    print(f"   ⚠️ Variability {status}")
    return False


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, forecast, param_name):
    mae = float(mean_absolute_error(actual, forecast))
    mse = float(mean_squared_error(actual, forecast))
    rmse = float(np.sqrt(mse))
    
    if is_rainfall(param_name):
        rain_mask = actual >= 1.0
        if np.sum(rain_mask) > 0:
            mape = float(np.mean(np.abs(
                (actual[rain_mask] - forecast[rain_mask]) / actual[rain_mask]
            )) * 100)
        else:
            mape = 0.0
        mae_ratio = (mae / (np.mean(actual) + 1e-6)) * 100
    else:
        non_zero_mask = np.abs(actual) > 0.1
        if np.sum(non_zero_mask) > 0:
            mape = float(np.mean(np.abs(
                (actual[non_zero_mask] - forecast[non_zero_mask]) / actual[non_zero_mask]
            )) * 100)
        else:
            mape = float('inf')
        mae_ratio = None
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mae_ratio': mae_ratio}


# ============================================================
# GRID CONFIG
# ============================================================

def get_grid_config(param_name):
    """Get hyperparameter grid untuk setiap tipe parameter"""
    
    if is_rainfall(param_name):
        # Grid search untuk rainfall
        return {
            'hidden_sizes': [64, 128],
            'num_layers_options': [2, 3],
            'learning_rates': [0.001, 0.003],
            'seq_lengths': [14, 21],
            'dropout_rates': [0.2, 0.3],
            'batch_sizes': [32],
            'max_epochs': 80,
            'patience': 10
        }
    elif is_ndvi(param_name):
        return {
            'hidden_sizes': [32, 64],
            'num_layers_options': [1, 2],
            'learning_rates': [0.001, 0.005],
            'seq_lengths': [3, 5],
            'dropout_rates': [0.1, 0.2],
            'batch_sizes': [16],
            'max_epochs': 100,
            'patience': 15
        }
    elif param_name in TEMP_PARAMS:
        return {
            'hidden_sizes': [64, 128],
            'num_layers_options': [2],
            'learning_rates': [0.001],
            'seq_lengths': [14, 21],
            'dropout_rates': [0.1, 0.2],
            'batch_sizes': [32],
            'max_epochs': 80,
            'patience': 10
        }
    else:
        return {
            'hidden_sizes': [64, 128],
            'num_layers_options': [2],
            'learning_rates': [0.001],
            'seq_lengths': [21, 30],
            'dropout_rates': [0.2, 0.3],
            'batch_sizes': [32],
            'max_epochs': 80,
            'patience': 8
        }


# ============================================================
# LSTM TRAINING
# ============================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, hidden_size, num_layers, 
                       dropout, lr, batch_size, max_epochs, patience,
                       train_scaled, seq_len, scaler, decomposition,
                       split_point, train_data, param_name, transform_params=None):
    global use_cuda
    current_device = torch.device("cuda" if use_cuda else "cpu")
    
    model = LSTMModel(1, hidden_size, num_layers, dropout).to(current_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.from_numpy(X_train).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(y_train).float().unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_epochs = 0
    
    for _ in range(max_epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(current_device), batch_y.to(current_device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        actual_epochs += 1
        
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1).to(current_device)
            y_val_t = torch.from_numpy(y_val).float().unsqueeze(-1).to(current_device)
            val_loss = criterion(model(X_val_t), y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    # Evaluate dengan recursive forecasting
    model.eval()
    forecast_scaled = []
    current_seq = train_scaled[-seq_len:].copy()
    
    with torch.no_grad():
        for _ in range(len(y_val) + seq_len):
            seq_t = torch.from_numpy(current_seq).float().unsqueeze(0).unsqueeze(-1).to(current_device)
            pred = model(seq_t).cpu().item()
            forecast_scaled.append(pred)
            current_seq = np.append(current_seq[1:], pred)
    
    forecast_residual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    
    # Reconstruct dengan seasonal + trend
    if decomposition is not None:
        val_len = len(y_val) + seq_len
        val_seasonal = decomposition['seasonal'][split_point:split_point + val_len]
        val_trend = decomposition['trend'][split_point:split_point + val_len]
        min_len = min(len(forecast_residual), len(val_seasonal), len(val_trend))
        forecast = forecast_residual[:min_len] + val_seasonal[:min_len] + val_trend[:min_len]
    else:
        forecast = forecast_residual
    
    # Post-process (termasuk inverse transform untuk rainfall)
    forecast = post_process_forecast(forecast, param_name, train_data.values, transform_params)
    
    # Calculate metrics
    actual = train_data.iloc[split_point:split_point + len(forecast)].values
    min_len = min(len(actual), len(forecast))
    metrics = calculate_metrics(actual[:min_len], forecast[:min_len], param_name)
    
    return metrics, actual_epochs


def grid_search_lstm_params(train_data, param_name, validation_ratio=0.10):
    """Grid search untuk SEMUA parameters termasuk rainfall"""
    global use_cuda
    
    print(f"\n{'='*60}")
    print(f"🔍 GRID SEARCH: {param_name}")
    print(f"{'='*60}")
    
    min_length = 46 if is_ndvi(param_name) else 365
    if len(train_data) < min_length:
        print(f"❌ Insufficient data ({len(train_data)} < {min_length})")
        return None, None, None, None
    
    # Transform untuk rainfall
    transform_params = None
    if is_rainfall(param_name):
        transformed_data, transform_params = apply_log_transform(train_data.values, param_name)
        working_series = pd.Series(transformed_data, index=train_data.index)
    else:
        working_series = train_data
    
    # Decomposition
    decomposition = seasonal_decompose_data(working_series, param_name)
    
    if decomposition is not None:
        working_data = pd.Series(decomposition['residual'], index=working_series.index).ffill().bfill()
        print(f"✓ Training on RESIDUAL (std: {working_data.std():.4f})")
    else:
        working_data = working_series
    
    # Train/validation split
    val_size = max(4 if is_ndvi(param_name) else 30, int(len(working_data) * validation_ratio))
    split_point = len(working_data) - val_size
    train_split = working_data[:split_point]
    val_split = working_data[split_point:]
    
    print(f"Train: {len(train_split)}, Validation: {len(val_split)}")
    
    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_split.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_split.values.reshape(-1, 1)).flatten()
    
    # Grid config
    config = get_grid_config(param_name)
    total_combos = (
        len(config['hidden_sizes']) * len(config['num_layers_options']) *
        len(config['learning_rates']) * len(config['seq_lengths']) *
        len(config['dropout_rates']) * len(config['batch_sizes'])
    )
    
    print(f"\n📋 Grid: {total_combos} combinations")
    print(f"{'='*60}\n")
    
    best_aic = float('inf')
    best_params = None
    best_metrics = None
    combo = 0
    
    for seq_len in config['seq_lengths']:
        if len(train_split) < seq_len + 1:
            continue
        
        X_train, y_train = create_sequences(train_scaled, seq_len)
        X_val, y_val = create_sequences(val_scaled, seq_len)
        
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        
        for batch_size in config['batch_sizes']:
            for hidden_size in config['hidden_sizes']:
                for num_layers in config['num_layers_options']:
                    for dropout in config['dropout_rates']:
                        for lr in config['learning_rates']:
                            combo += 1
                            print(f"[{combo}/{total_combos}] seq={seq_len}, h={hidden_size}, l={num_layers}, d={dropout}, lr={lr}", end=" ")
                            
                            try:
                                result = train_and_evaluate(
                                    X_train, y_train, X_val, y_val,
                                    hidden_size, num_layers, dropout, lr, batch_size,
                                    config['max_epochs'], config['patience'],
                                    train_scaled, seq_len, scaler, decomposition,
                                    split_point, train_data, param_name, transform_params
                                )
                                
                                if result is None:
                                    continue
                                
                                metrics, epochs = result
                                aic = calculate_aic(
                                    metrics['mse'], 
                                    LSTMModel(1, hidden_size, num_layers, dropout).count_parameters(),
                                    len(val_split)
                                )
                                
                                if is_rainfall(param_name):
                                    print(f"→ ep={epochs}, AIC={aic:.0f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
                                else:
                                    print(f"→ ep={epochs}, AIC={aic:.0f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.1f}%")
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_params = {
                                        'hidden_size': hidden_size,
                                        'num_layers': num_layers,
                                        'learning_rate': lr,
                                        'seq_length': seq_len,
                                        'dropout': dropout,
                                        'batch_size': batch_size,
                                        'epochs': epochs
                                    }
                                    best_metrics = {**metrics, 'aic': aic}
                                    print(f"   ✅ NEW BEST!")
                                    
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower() and use_cuda:
                                    print(f"→ OOM, switching to CPU")
                                    use_cuda = False
                                else:
                                    print(f"→ Error: {str(e)[:50]}")
                            finally:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    if best_params:
        print(f"🎯 BEST MODEL:")
        print(f"   AIC: {best_aic:.2f}, MAE: {best_metrics['mae']:.4f}, MAPE: {best_metrics['mape']:.2f}%")
        print(f"   Params: {best_params}")
    else:
        print(f"❌ No valid model found")
    print(f"{'='*60}\n")
    
    return best_params, best_metrics, decomposition, transform_params


def fit_lstm_model(data, best_params, param_name):
    global use_cuda
    
    print(f"\n🧪 Fitting final model...")
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(data_scaled, best_params['seq_length'])
    if len(X) == 0:
        return None, None
    
    current_device = torch.device("cuda" if use_cuda else "cpu")
    model = LSTMModel(1, best_params['hidden_size'], best_params['num_layers'], 
                      best_params['dropout']).to(current_device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(y).float().unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    for _ in range(best_params['epochs']):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(current_device), batch_y.to(current_device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    print(f"✓ Model fitted ({best_params['epochs']} epochs)")
    return model, scaler


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast", 
                      config_id=None, append_column_id=True, client=None, start_date=None, end_date=None):
    global use_cuda
    
    print(f"\n{'#'*60}")
    print(f"# LSTM ANALYSIS: {collection_name}.{target_column}")
    print(f"{'#'*60}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    should_close = client is None
    if client is None:
        load_dotenv()
        client = MongoClient(os.getenv("MONGODB_URI"))
    
    db = client["tugas_akhir"]
    
    try:
        source_data = list(db[collection_name].find().sort("Date", 1))
        print(f"📥 Fetched {len(source_data)} records from {collection_name}")
        
        if not source_data:
            raise ValueError(f"No data in {collection_name}")
        
        df = pd.DataFrame(source_data)
        
        date_col = next((c for c in ['Date', 'date', 'timestamp', 'Timestamp'] if c in df.columns), None)
        if date_col is None:
            raise ValueError("No date column found")
        
        df['timestamp'] = pd.to_datetime(df[date_col])
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        freq = '16D' if is_ndvi(target_column) else 'D'
        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        df = df.reindex(date_range).interpolate(method='linear')
        
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found")
        
        param_data = df[target_column].dropna()
        
        if len(param_data) < 100:
            raise ValueError(f"Insufficient data: {len(param_data)} < 100")
        
        print(f"📊 Data: {len(param_data)} values, mean={param_data.mean():.3f}, std={param_data.std():.3f}")
        print(f"   Column: {target_column}")
        print(f"   Collection: {collection_name}")
        
        # ============================================================
        # GRID SEARCH
        # ============================================================
        best_params, error_metrics, decomposition, transform_params = grid_search_lstm_params(
            param_data, target_column
        )
        
        if best_params is None:
            raise ValueError("No valid model found")
        
        # ============================================================
        # SAVE DECOMPOSITION DATA (PERBAIKAN DI SINI)
        # ============================================================
        if decomposition is not None:
            print(f"\n{'='*60}")
            print(f"🔍 DECOMPOSITION INFO FOR {target_column}")
            print(f"{'='*60}")
            
            # PENTING: Decompose dilakukan pada data SEBELUM transform
            # Tapi jika rainfall, decomposition sudah pada transformed data
            # Jadi kita perlu decompose data ASLI untuk save
            
            if is_rainfall(target_column) and transform_params:
                print(f"   ℹ️ Rainfall detected - decomposing ORIGINAL data (not log-transformed)")
                # Decompose ulang dari data asli (bukan log-transformed)
                decomposition_original = seasonal_decompose_data(param_data, target_column)
                
                if decomposition_original is not None:
                    decompose_count = save_decompose_data(
                        df_index=param_data.index,
                        decomposition=decomposition_original,
                        param_name=target_column,
                        config_id=config_id,
                        collection_name="decompose-lstm-temp",
                        client=client,
                        original_data=param_data.values
                    )
                else:
                    print(f"   ⚠️ Failed to decompose original data")
                    decompose_count = 0
            else:
                # Untuk non-rainfall, gunakan decomposition yang sudah ada
                decompose_count = save_decompose_data(
                    df_index=param_data.index,
                    decomposition=decomposition,
                    param_name=target_column,
                    config_id=config_id,
                    collection_name="decompose-lstm-temp",
                    client=client,
                    original_data=param_data.values
                )
            
            print(f"✅ Saved {decompose_count} decompose documents for {target_column}")
        
        # Prepare data untuk final model
        if is_rainfall(target_column) and transform_params:
            transformed_data, _ = apply_log_transform(param_data.values, target_column)
            working_series = pd.Series(transformed_data, index=param_data.index)
        else:
            working_series = param_data
        
        # Fit final model
        if decomposition is not None:
            residual_data = pd.Series(decomposition['residual'], index=working_series.index).ffill().bfill()
            final_model, scaler = fit_lstm_model(residual_data, best_params, target_column)
        else:
            final_model, scaler = fit_lstm_model(working_series, best_params, target_column)
        
        if final_model is None:
            raise ValueError("Failed to fit model")
        
        # Setup forecast dates
        # SMART AUTO-ADJUSTMENT WITH AUTO-DETECT BACKTEST
        last_historical_date = param_data.index[-1]
        print(f"\n📊 Last historical data: {last_historical_date.date()}")
        
        gap_warning = None
        
        if start_date is not None and end_date is not None:
            user_start = pd.to_datetime(start_date)
            user_end = pd.to_datetime(end_date)
            requested_duration = (user_end - user_start).days
            
            print(f"   User requested: {user_start.date()} to {user_end.date()} ({requested_duration + 1} days)")
            
            # ============================================================
            # AUTO-DETECT BACKTEST: User start < last historical
            # ============================================================
            if user_start < last_historical_date:
                # Hitung overlap dengan historical data
                if user_end <= last_historical_date:
                    # Full backtest (seluruh range ada historical data)
                    backtest_days = (user_end - user_start).days + 1
                    true_forecast_days = 0
                    mode = "FULL_BACKTEST"
                    
                    print(f"\n🔬 AUTO-DETECTED: FULL BACKTEST MODE")
                    print(f"   → All {backtest_days} days have historical data")
                    print(f"   → This is for validation/evaluation purposes")
                    
                else:
                    # Hybrid mode (sebagian backtest, sebagian forecast)
                    backtest_days = (last_historical_date - user_start).days + 1
                    true_forecast_days = (user_end - last_historical_date).days
                    mode = "HYBRID_BACKTEST"
                    
                    print(f"\n🔬 AUTO-DETECTED: HYBRID MODE")
                    print(f"   → Backtest: {backtest_days} days ({user_start.date()} to {last_historical_date.date()})")
                    print(f"   → True forecast: {true_forecast_days} days ({(last_historical_date + pd.Timedelta(days=1)).date()} to {user_end.date()})")
                
                # Use user dates as-is untuk backtest
                forecast_start_date = user_start
                forecast_end_date = user_end
                
                print(f"   ✅ Proceeding with user-specified dates for backtest")
                print(f"   → You can compare forecast vs actual data for validation")
                
                gap_warning = {
                    "type": mode,
                    "start": forecast_start_date.date(),
                    "end": forecast_end_date.date(),
                    "backtest_days": backtest_days,
                    "true_forecast_days": true_forecast_days,
                    "last_historical": last_historical_date.date(),
                    "reason": "Auto-detected backtest mode. Historical data available for validation."
                }
            
            # ============================================================
            # KASUS 2: User start = last historical + 1 (Perfect continuity)
            # ============================================================
            elif user_start == last_historical_date + pd.Timedelta(days=1):
                print(f"\n✅ CONTINUOUS FORECAST:")
                print(f"   Last historical: {last_historical_date.date()}")
                print(f"   Forecast start: {user_start.date()}")
                print(f"   → Perfect continuity! Optimal accuracy expected")
                
                forecast_start_date = user_start
                forecast_end_date = user_end
                
                gap_warning = {
                    "type": "CONTINUOUS",
                    "start": forecast_start_date.date(),
                    "end": forecast_end_date.date(),
                    "gap_days": 0,
                    "reason": "Data is continuous. Optimal accuracy expected."
                }
            
            # ============================================================
            # KASUS 3: User start > last historical (Ada gap)
            # ============================================================
            else:
                gap_days = (user_start - last_historical_date).days - 1
                
                if gap_days > 0:
                    print(f"\n⚠️  GAP DETECTED: {gap_days} days")
                    print(f"   Last historical: {last_historical_date.date()}")
                    print(f"   User start: {user_start.date()}")
                    
                    # Decompose data terlebih dahulu untuk gap filling (jika belum)
                    if decomposition is None:
                        temp_decomposition = seasonal_decompose_data(param_data, target_column)
                    else:
                        temp_decomposition = decomposition
                    
                    # FILL GAP WITH SYNTHETIC DATA
                    param_data, gap_fill_info = fill_gap_with_synthetic_data(
                        param_data=param_data,
                        last_historical_date=last_historical_date,
                        target_start_date=user_start,
                        param_name=target_column,
                        decomposition=temp_decomposition
                    )
                    
                    # Update last historical date (now includes synthetic)
                    new_last_date = param_data.index[-1]
                    
                    print(f"   ✅ Gap filled! Data now extends to: {new_last_date.date()}")
                    
                    # Re-decompose dengan data yang sudah extended (untuk rainfall perlu transform dulu)
                    print(f"\n🔄 Re-decomposing extended data...")
                    if is_rainfall(target_column) and transform_params:
                        extended_transformed, _ = apply_log_transform(param_data.values, target_column)
                        extended_series = pd.Series(extended_transformed, index=param_data.index)
                        decomposition = seasonal_decompose_data(extended_series, target_column)
                    else:
                        decomposition = seasonal_decompose_data(param_data, target_column)
                    
                    # Kategorisasi severity untuk informasi
                    if gap_days > 90:
                        severity = "HIGH"
                    elif gap_days > 30:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"
                    
                    gap_warning = {
                        "type": "GAP_FILLED",
                        "severity": severity,
                        "original_gap_days": gap_days,
                        "gap_fill_info": gap_fill_info,
                        "original_last_historical": last_historical_date.strftime("%Y-%m-%d"),
                        "extended_last_date": new_last_date.strftime("%Y-%m-%d"),
                        "start": user_start.strftime("%Y-%m-%d"),
                        "end": user_end.strftime("%Y-%m-%d"),
                        "reason": f"Gap of {gap_days} days filled with synthetic data using seasonal pattern matching."
                    }
                else:
                    gap_warning = {
                        "type": "CONTINUOUS",
                        "start": user_start.strftime("%Y-%m-%d"),
                        "end": user_end.strftime("%Y-%m-%d"),
                        "gap_days": 0,
                        "reason": "Data is continuous. Optimal accuracy expected."
                    }
                
                # Proceed with user dates
                forecast_start_date = user_start
                forecast_end_date = user_end
        else:
            # Default: mulai dari hari setelah data historis terakhir
            forecast_start_date = last_historical_date + pd.Timedelta(days=1)
            forecast_end_date = forecast_start_date + pd.Timedelta(days=364)
            gap_warning = {
                "type": "DEFAULT",
                "start": forecast_start_date.date(),
                "end": forecast_end_date.date(),
                "reason": "Default 365-day forecast from last historical date."
            }
            print(f"\n📅 Default forecast: {forecast_start_date.date()} to {forecast_end_date.date()} (365 days)")
        
        freq = '16D' if is_ndvi(target_column) else 'D'
        forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq=freq)
        forecast_steps = len(forecast_dates)
        
        print(f"\n🔮 Generating forecast: {forecast_steps} steps ({forecast_start_date.date()} to {forecast_end_date.date()})")
        
        # ============================================================
        # RE-PREPARE DATA AFTER GAP FILLING (jika ada gap yang di-fill)
        # ============================================================
        if gap_warning and gap_warning.get('type') == 'GAP_FILLED':
            print(f"\n🔄 Re-preparing data with extended dataset...")
            
            # Re-transform jika rainfall
            if is_rainfall(target_column) and transform_params:
                transformed_data, _ = apply_log_transform(param_data.values, target_column)
                working_series = pd.Series(transformed_data, index=param_data.index)
            else:
                working_series = param_data
            
            # Re-fit model dengan extended data
            if decomposition is not None:
                residual_data = pd.Series(decomposition['residual'], index=working_series.index).ffill().bfill()
                final_model, scaler = fit_lstm_model(residual_data, best_params, target_column)
            else:
                final_model, scaler = fit_lstm_model(working_series, best_params, target_column)
            
            if final_model is None:
                raise ValueError("Failed to re-fit model with extended data")
            
            print(f"   ✅ Model re-fitted with extended data ({len(param_data)} records)")
        
        # Generate forecast
        if decomposition is not None:
            working_data = pd.Series(decomposition['residual'], index=working_series.index).ffill().bfill()
        else:
            working_data = working_series
        
        data_scaled = scaler.transform(working_data.values.reshape(-1, 1)).flatten()
        current_seq = data_scaled[-best_params['seq_length']:].copy()
        
        final_model.eval()
        forecast_scaled = []
        current_device = torch.device("cuda" if use_cuda else "cpu")
        
        with torch.no_grad():
            for step in range(forecast_steps):
                seq_t = torch.from_numpy(current_seq).float().unsqueeze(0).unsqueeze(-1).to(current_device)
                pred = final_model(seq_t).cpu().item()
                forecast_scaled.append(pred)
                current_seq = np.append(current_seq[1:], pred)
                
                if (step + 1) % 100 == 0:
                    print(f"   Generated {step + 1}/{forecast_steps}...")
        
        # Inverse scale
        forecast_residual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        
        # Reconstruct
        if decomposition is not None:
            print(f"\n📊 Reconstructing...")
            forecast_seasonal = extrapolate_seasonal(
                decomposition['seasonal'], 
                forecast_steps, 
                decomposition['period'],
                param_name=target_column
            )
            forecast_trend = extrapolate_trend(decomposition['trend'], forecast_steps, target_column)
            forecast = forecast_residual + forecast_seasonal + forecast_trend
            
            print(f"   Residual: [{forecast_residual.min():.3f}, {forecast_residual.max():.3f}]")
            print(f"   Seasonal: [{forecast_seasonal.min():.3f}, {forecast_seasonal.max():.3f}]")
            print(f"   Trend: [{forecast_trend.min():.3f}, {forecast_trend.max():.3f}]")
        else:
            forecast = forecast_residual
        
        # Post-process
        forecast = post_process_forecast(forecast, target_column, param_data.values, transform_params)
        
        # Variability check
        check_variability(forecast, param_data, target_column)
        
        print(f"\n✓ Forecast: [{forecast.min():.3f}, {forecast.max():.3f}], mean={forecast.mean():.3f}")
        
        # Save to MongoDB
        print(f"\n💾 Saving to {save_collection}...")
        
        upsert_count = 0
        for i, forecast_date in enumerate(forecast_dates):
            value = float(forecast[i])
            if np.isnan(value) or np.isinf(value):
                continue
            
            doc = {
                "forecast_date": forecast_date.to_pydatetime(),
                "timestamp": datetime.now().isoformat(),
                "source_collection": collection_name,
                "config_id": config_id,
                "parameters": {
                    target_column: {
                        "forecast_value": value,
                        "model_metadata_lstm": {
                            **best_params,
                            "decomposition_used": decomposition is not None,
                            "transform_method": transform_params['method'] if transform_params else None
                        }
                    }
                }
            }
            
            if append_column_id:
                doc["column_id"] = f"{collection_name}_{target_column}"
            
            result = db[save_collection].update_one(
                {"forecast_date": doc["forecast_date"], "config_id": config_id},
                {"$set": {
                    f"parameters.{target_column}": doc["parameters"][target_column],
                    "timestamp": doc["timestamp"],
                    "source_collection": collection_name,
                    "column_id": doc.get("column_id")
                }},
                upsert=True
            )
            if result.upserted_id or result.modified_count > 0:
                upsert_count += 1
        
        print(f"✓ Saved {upsert_count} documents")
        
        result_summary = {
            "collection_name": collection_name,
            "target_column": target_column,
            "forecast_days": len(forecast_dates),
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_params": best_params,
            "error_metrics": convert_to_python_types(error_metrics) if error_metrics else None,
            "decomposition_used": decomposition is not None,
            "transform_params": transform_params,
            "forecast_range": {
                "start": forecast_start_date.strftime("%Y-%m-%d"),
                "end": forecast_end_date.strftime("%Y-%m-%d"),
                "min": float(forecast.min()),
                "max": float(forecast.max()),
                "mean": float(forecast.mean()),
                "std": float(forecast.std())
            },
            "gap_warning": convert_to_python_types(gap_warning) if gap_warning else None  # ← TAMBAHAN
        }

        print(f"\n{'#'*60}")
        print(f"# ✅ COMPLETED: {collection_name}.{target_column}")
        print(f"{'#'*60}\n")

        # ============================================================
        # BACKTEST VALIDATION (jika applicable)
        # ============================================================
        if gap_warning and gap_warning.get('type') in ['FULL_BACKTEST', 'HYBRID_BACKTEST']:
            print(f"\n🔬 BACKTEST VALIDATION:")
            
            backtest_days = gap_warning.get('backtest_days', 0)
            if backtest_days > 0:
                # Ambil actual data untuk backtest period
                backtest_forecast = forecast[:backtest_days]
                
                # Get actual data from historical
                backtest_start = forecast_dates[0]
                backtest_end = forecast_dates[backtest_days - 1]
                
                actual_data = param_data.loc[backtest_start:backtest_end].values
                
                if len(actual_data) == len(backtest_forecast):
                    # Calculate validation metrics
                    val_metrics = calculate_metrics(actual_data, backtest_forecast, target_column)
                    
                    print(f"   Backtest period: {backtest_start.date()} to {backtest_end.date()}")
                    print(f"   Backtest MAE: {val_metrics['mae']:.4f}")
                    print(f"   Backtest MAPE: {val_metrics['mape']:.2f}%")
                    print(f"   Backtest RMSE: {val_metrics['rmse']:.4f}")
                    
                    # Update gap_warning dengan backtest metrics
                    result_summary['gap_warning']['backtest_metrics'] = convert_to_python_types(val_metrics)  # ← TAMBAHAN
                else:
                    print(f"   ⚠️ Length mismatch: actual={len(actual_data)}, forecast={len(backtest_forecast)}")
        
        return result_summary
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if should_close:
            client.close()


def save_decompose_data(df_index, decomposition, param_name, config_id, collection_name, client, original_data=None):
    """Save decomposition data to MongoDB with detailed debugging"""
    print(f"\n{'='*60}")
    print(f"💾 SAVING DECOMPOSE DATA: {param_name}")
    print(f"{'='*60}")
    
    if decomposition is None:
        print(f"   ⚠️ No decomposition to save for {param_name}")
        return 0
    
    db = client["tugas_akhir"]
    saved_count = 0
    
    # Siapkan data decompose
    trend = decomposition['trend']
    seasonal = decomposition['seasonal']
    residual = decomposition['residual']
    
    print(f"📊 Decompose Stats for {param_name}:")
    print(f"   Data length: {len(df_index)}")
    print(f"   Trend length: {len(trend)}")
    print(f"   Seasonal length: {len(seasonal)}")
    print(f"   Residual length: {len(residual)}")
    print(f"   Date range: {df_index[0]} to {df_index[-1]}")
    print(f"   Trend range: [{np.nanmin(trend):.3f}, {np.nanmax(trend):.3f}]")
    print(f"   Seasonal range: [{np.nanmin(seasonal):.3f}, {np.nanmax(seasonal):.3f}]")
    print(f"   Residual range: [{np.nanmin(residual):.3f}, {np.nanmax(residual):.3f}]")
    
    if original_data is not None:
        print(f"   Original data range: [{original_data.min():.3f}, {original_data.max():.3f}]")
    
    # Verify lengths match
    if len(df_index) != len(trend) or len(df_index) != len(seasonal) or len(df_index) != len(residual):
        print(f"   ❌ ERROR: Length mismatch!")
        print(f"      Index: {len(df_index)}, Trend: {len(trend)}, Seasonal: {len(seasonal)}, Residual: {len(residual)}")
        return 0
    
    # Group by date untuk menggabungkan parameter
    decompose_data = {}
    nan_count = 0
    
    for i, date in enumerate(df_index):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        
        # Skip jika nilai NaN
        if np.isnan(trend[i]) or np.isnan(seasonal[i]) or np.isnan(residual[i]):
            nan_count += 1
            continue
        
        if date_str not in decompose_data:
            decompose_data[date_str] = {
                "date": date_str,
                "timestamp": datetime.now().isoformat(),
                "config_id": config_id,
                "parameters": {}
            }
        
        decompose_data[date_str]["parameters"][param_name] = {
            "trend": float(trend[i]),
            "seasonal": float(seasonal[i]),
            "resid": float(residual[i])
        }
    
    if nan_count > 0:
        print(f"   ⚠️ Skipped {nan_count} NaN values")
    
    print(f"\n📝 Prepared {len(decompose_data)} unique dates for {param_name}")
    
    # Show first 3 entries untuk debug
    sample_dates = list(decompose_data.keys())[:3]
    print(f"\n🔍 Sample entries (first 3):")
    for date_str in sample_dates:
        doc = decompose_data[date_str]
        if param_name in doc["parameters"]:
            params = doc["parameters"][param_name]
            print(f"   {date_str}: trend={params['trend']:.3f}, seasonal={params['seasonal']:.3f}, resid={params['resid']:.3f}")
    
    # Upsert ke MongoDB
    print(f"\n💾 Upserting to {collection_name}...")
    for date_str, doc in decompose_data.items():
        result = db[collection_name].update_one(
            {"date": date_str, "config_id": config_id},
            {"$set": {
                f"parameters.{param_name}": doc["parameters"][param_name],
                "timestamp": doc["timestamp"]
            }},
            upsert=True
        )
        if result.upserted_id or result.modified_count > 0:
            saved_count += 1
    
    print(f"✅ Successfully saved {saved_count} decompose records for {param_name}")
    print(f"{'='*60}\n")
    
    return saved_count


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    result = run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
    print(f"\nResult: {result}")