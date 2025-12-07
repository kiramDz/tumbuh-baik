"""
LSTM Dynamic Forecasting with Seasonal Decomposition
Version: 2.3 (With Grid Search for All Parameters)
"""

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

RAINFALL_PARAMS = ["RR", "RR_imputed"]
NDVI_PARAMS = ["NDVI", "NDVI_imputed"]
TEMP_PARAMS = ["TAVG", "TMAX", "TMIN"]
HUMIDITY_PARAMS = ["RH_AVG", "RH_AVG_preprocessed"]
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
    # Clip untuk mencegah overflow
    data_clipped = np.clip(data_array, -10, 6)
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
    """Extrapolate seasonal pattern dengan dampening untuk rainfall"""
    one_cycle = seasonal_pattern[-period:]
    n_repeats = (forecast_steps // period) + 2
    extended = np.tile(one_cycle, n_repeats)[:forecast_steps]
    
    # Dampening untuk rainfall - PERBAIKAN: kurangi ketatnya
    if param_name is not None and is_rainfall(param_name):
        seasonal_std = np.std(one_cycle)
        seasonal_mean = np.mean(one_cycle)
        
        # UBAH: dari 1.0 std ke 2.0 std agar lebih variabel
        lower_bound = seasonal_mean - 2.0 * seasonal_std
        upper_bound = seasonal_mean + 2.0 * seasonal_std
        extended = np.clip(extended, lower_bound, upper_bound)
        
        # UBAH: cap maksimum dari 3.5 ke 4.5 (log1p(90) ≈ 4.5)
        extended = np.clip(extended, -2.0, 4.5)
        
        print(f"   📉 Seasonal dampened: [{extended.min():.3f}, {extended.max():.3f}]")
    
    return extended


def extrapolate_trend(trend, forecast_steps):
    window = min(30, len(trend) // 4)
    last_values = trend[-window:]
    slope = (last_values[-1] - last_values[0]) / len(last_values) if len(last_values) > 1 else 0
    
    forecast_trend = []
    current_value = trend[-1]
    current_slope = slope
    damping = 0.98
    
    for _ in range(forecast_steps):
        current_value += current_slope
        current_slope *= damping
        forecast_trend.append(current_value)
    
    return np.array(forecast_trend)


# ============================================================
# POST PROCESSING
# ============================================================

def post_process_forecast(forecast, param_name, historical_data=None, transform_params=None):
    """Post-process forecast values"""
    forecast = np.array(forecast).flatten()
    
    if is_rainfall(param_name):
        if transform_params is not None:
            # UBAH: dari [-5, 5.5] ke [-5, 6.0] untuk allow nilai lebih tinggi
            forecast = np.clip(forecast, -5, 6.0)  # log1p(400) ≈ 6.0
            forecast = inverse_log_transform(forecast, transform_params, param_name)
            print(f"   ✓ Log1p inverse applied")
        
        forecast = np.where(forecast < 0.5, 0, forecast)
        
        if historical_data is not None:
            hist_max = float(np.max(historical_data))
            hist_p99 = float(np.percentile(historical_data, 99))
            hist_p95 = float(np.percentile(historical_data, 95))
            
            # UBAH: cap lebih longgar
            max_cap = min(hist_max * 1.2, hist_p99 * 2.0, 250)  # Lebih longgar
            print(f"   📊 Cap: hist_max={hist_max:.1f}, P99={hist_p99:.1f}, cap={max_cap:.1f}")
        else:
            max_cap = 200
        
        forecast = np.clip(forecast, 0, max_cap)
        forecast = np.round(forecast, 1)
    elif is_ndvi(param_name):
        forecast = np.clip(forecast, -1, 1)
    elif param_name in HUMIDITY_PARAMS:
        forecast = np.clip(forecast, 0, 100)
    elif param_name in TEMP_PARAMS:
        forecast = np.clip(forecast, 10, 50)
    elif param_name in SOLAR_PARAMS:
        forecast = np.clip(forecast, 0, 40)
    
    return forecast


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
# GRID CONFIG (termasuk rainfall)
# ============================================================

def get_grid_config(param_name):
    """Get hyperparameter grid untuk setiap tipe parameter"""
    
    if is_rainfall(param_name):
        # Grid search untuk rainfall
        return {
            'hidden_sizes': [32, 64],
            'num_layers_options': [1, 2],
            'learning_rates': [0.002, 0.005],
            'seq_lengths': [7, 14],
            'dropout_rates': [0.1, 0.2],
            'batch_sizes': [32],
            'max_epochs': 60,
            'patience': 8
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


def grid_search_lstm_params(train_data, param_name, validation_ratio=0.30):
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
        print(f"📥 Fetched {len(source_data)} records")
        
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
        
        # ============================================================
        # GRID SEARCH (untuk semua parameter termasuk rainfall)
        # ============================================================
        best_params, error_metrics, decomposition, transform_params = grid_search_lstm_params(
            param_data, target_column
        )
        
        if best_params is None:
            raise ValueError("No valid model found")
        
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
        if start_date is not None and end_date is not None:
            forecast_start = pd.to_datetime(start_date)
            forecast_end = pd.to_datetime(end_date)
            forecast_start_date = forecast_start  # ✅ TAMBAH INI
            forecast_end_date = forecast_end      # ✅ TAMBAH INI
            print(f"\n📅 Custom forecast range: {forecast_start.date()} to {forecast_end.date()}")
        else:
            last_data_date = param_data.index[-1]
            forecast_start_date = last_data_date + pd.Timedelta(days=1)
            forecast_end_date = forecast_start_date + pd.Timedelta(days=364)
            forecast_start = forecast_start_date
            forecast_end = forecast_end_date
            print(f"[INFO] Using default data range: {forecast_start_date.date()} to {forecast_end_date.date()}")
            forecast_start = forecast_start_date
            forecast_end = forecast_end_date
        
        freq = '16D' if is_ndvi(target_column) else 'D'
        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq=freq)
        forecast_steps = len(forecast_dates)
        
        print(f"\n🔮 Generating forecast: {forecast_steps} steps ({forecast_start.date()} to {forecast_end.date()})")
        
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
            forecast_trend = extrapolate_trend(decomposition['trend'], forecast_steps)
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
                        "model_metadata": {
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
            }
        }
        
        print(f"\n{'#'*60}")
        print(f"# ✅ COMPLETED: {collection_name}.{target_column}")
        print(f"{'#'*60}\n")
        
        return result_summary
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if should_close:
            client.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    result = run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
    print(f"\nResult: {result}")