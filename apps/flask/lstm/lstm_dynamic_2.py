from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMForecaster:
    def __init__(self, sequence_length=365, epochs=100, batch_size=32):
        """
        Initialize LSTM Forecaster
        
        Args:
            sequence_length (int): Number of time steps to look back (default: 365 days)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        
    def create_sequences(self, data):
        """
        Create sequences for LSTM training
        
        Args:
            data (array): Time series data
            
        Returns:
            X, y: Input sequences and target values
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture (similar to notebook style)
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            model: Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1)  # Output layer untuk satu langkah prediksi
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, train_data, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            train_data (array): Training time series data
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            history: Training history
        """
        print(f"Training LSTM model with {len(train_data)} data points...")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError(f"Insufficient data for sequence length {self.sequence_length}")
        
        print(f"Created {len(X)} sequences for training")
        
        # Reshape for LSTM input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self.build_model((X.shape[1], 1))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print("✓ LSTM model training completed")
        
        return history
    
    def predict(self, data, steps=1):
        """
        Generate forecasts
        
        Args:
            data (array): Input data for prediction
            steps (int): Number of steps to forecast
            
        Returns:
            array: Forecast values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale the input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Take the last sequence_length points for prediction
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Prepare input for prediction
            X_pred = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Store prediction
            predictions.append(pred_scaled)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def evaluate(self, test_data, test_targets):
        """
        Evaluate model performance
        
        Args:
            test_data (array): Test input data
            test_targets (array): True target values
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = []
        
        for i in range(len(test_targets)):
            # Use sequence ending at position i+sequence_length
            end_idx = i + self.sequence_length
            if end_idx > len(test_data):
                break
                
            input_data = test_data[:end_idx]
            pred = self.predict(input_data, steps=1)[0]
            predictions.append(pred)
        
        if len(predictions) == 0:
            return None
        
        # Align predictions with targets
        aligned_targets = test_targets[:len(predictions)]
        predictions = np.array(predictions)
        
        # Calculate metrics
        mae = mean_absolute_error(aligned_targets, predictions)
        mse = mean_squared_error(aligned_targets, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((aligned_targets - predictions) / np.where(aligned_targets != 0, aligned_targets, 1))) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions,
            'targets': aligned_targets
        }

def post_process_forecast(forecast, param_name):
    """
    Post-processing untuk memastikan forecast masuk akal
    """
    if param_name == "RR" or param_name == "RR_imputed":  # Curah Hujan
        forecast = np.maximum(forecast, 0)
        forecast = np.minimum(forecast, 300)
        
    elif param_name == "RH_AVG":  # Kelembapan
        forecast = np.clip(forecast, 0, 100)
    
    elif param_name == "NDVI":  # Normalized Difference Vegetation Index
        forecast = np.clip(forecast, -1, 1)
    
    elif "Suhu" in param_name or "Temperature" in param_name:  # Suhu
        forecast = np.clip(forecast, -50, 60)
    
    return forecast

def optimize_lstm_params(train_data, param_name, validation_split=0.2):
    """
    Optimize LSTM hyperparameters using grid search (with longer sequence lengths)
    
    Args:
        train_data (array): Training data
        param_name (str): Parameter name for post-processing
        validation_split (float): Validation split ratio
        
    Returns:
        tuple: Best parameters and metrics
    """
    print(f"\n--- LSTM Hyperparameter Optimization for {param_name} ---")
    
    data_length = len(train_data)
    print(f"Data length: {data_length}")
    
    # Minimum requirement for LSTM
    if data_length < 500:
        print("❌ Insufficient data for LSTM optimization (minimum 500 points required)")
        return None, None
    
    # Dynamic sequence lengths based on data availability
    if data_length >= 2000:
        sequence_lengths = [365, 700, 1000]  # Like notebook: 700 days
        print("Using long sequence lengths (365, 700, 1000 days)")
    elif data_length >= 1500:
        sequence_lengths = [365, 500, 700]
        print("Using medium-long sequence lengths (365, 500, 700 days)")
    elif data_length >= 1000:
        sequence_lengths = [180, 365, 500]
        print("Using medium sequence lengths (180, 365, 500 days)")
    else:
        sequence_lengths = [90, 180, 365]
        print("Using shorter sequence lengths (90, 180, 365 days)")
    
    # Filter sequence lengths that are feasible
    max_seq_len = int(data_length * 0.7)  # Use max 70% of data for sequence
    sequence_lengths = [seq for seq in sequence_lengths if seq <= max_seq_len]
    
    if not sequence_lengths:
        print("❌ No feasible sequence lengths found")
        return None, None
    
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    batch_sizes = [16, 32, 64]
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    
    # Split data for validation
    split_point = int(len(train_data) * (1 - validation_split))
    train_split = train_data[:split_point]
    val_split = train_data[split_point:]
    
    print(f"Train split: {len(train_split)}, Validation split: {len(val_split)}")
    
    for seq_len in sequence_lengths:
        # Check if we have enough data for this sequence length
        if len(train_split) <= seq_len:
            print(f"Skipping seq_len={seq_len} (insufficient training data)")
            continue
            
        for batch_size in batch_sizes:
            try:
                print(f"\nTesting: sequence_length={seq_len}, batch_size={batch_size}")
                
                # Create LSTM forecaster
                forecaster = LSTMForecaster(
                    sequence_length=seq_len,
                    epochs=50,  # Reduced for optimization
                    batch_size=batch_size
                )
                
                # Train model
                history = forecaster.fit(train_split, validation_split=0.2)
                
                # Evaluate on validation set
                metrics = forecaster.evaluate(train_data[:split_point + seq_len], val_split)
                
                if metrics is None:
                    print(f"❌ Evaluation failed for seq_len={seq_len}, batch_size={batch_size}")
                    continue
                
                rmse = metrics['rmse']
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        'sequence_length': seq_len,
                        'batch_size': batch_size,
                        'epochs': 100  # Use more epochs for final model
                    }
                    best_metrics = metrics
                    print(f"✓ New best: seq_len={seq_len}, batch_size={batch_size}")
                    print(f"  RMSE={rmse:.3f}, MAE={metrics['mae']:.3f}, MAPE={metrics['mape']:.3f}%")
                
            except Exception as e:
                print(f"❌ Error for seq_len={seq_len}, batch_size={batch_size}: {str(e)}")
                continue
    
    if best_params is None:
        # Use default parameters with appropriate sequence length
        default_seq_len = min(365, max_seq_len)
        best_params = {
            'sequence_length': default_seq_len,
            'batch_size': 32,
            'epochs': 100
        }
        print(f"Using default parameters with sequence_length={default_seq_len}")
    
    print(f"✓ Best parameters: {best_params}")
    return best_params, best_metrics

def run_lstm_analysis(collection_name, target_column, save_collection="lstm-forecast", config_id=None, append_column_id=True, client=None):
    """
    Fungsi LSTM yang dinamis berdasarkan parameter dari forecast_config
    """
    print(f"=== Start LSTM Analysis for {collection_name}.{target_column} ===")
    
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
        
        # Pastikan indeks harian tanpa duplikasi
        df = df[~df.index.duplicated(keep='first')]
        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
        df = df.reindex(date_range, method='ffill')
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Cek apakah target column ada
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {collection_name}")
        
        # Get data (tanpa preprocessing karena data sudah bersih)
        param_data = df[target_column].dropna()
        
        if len(param_data) < 500:
            raise ValueError(f"Insufficient data for LSTM analysis: {len(param_data)} records (minimum 500 required)")
        
        # Convert to numpy array
        data_values = param_data.values
        
        # Optimize hyperparameters
        best_params, optimization_metrics = optimize_lstm_params(data_values, target_column)
        
        if best_params is None:
            raise ValueError(f"No valid LSTM parameters found for {target_column}")
        
        # Train final model with best parameters
        print(f"\nTraining final LSTM model with best parameters...")
        print(f"Using sequence_length: {best_params['sequence_length']} days")
        
        final_forecaster = LSTMForecaster(
            sequence_length=best_params['sequence_length'],
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size']
        )
        
        # Split data for final training (use 90% for training like notebook)
        train_size = int(len(data_values) * 0.9)
        train_data = data_values[:train_size]
        test_data = data_values[train_size:]
        
        # Train final model
        history = final_forecaster.fit(train_data, validation_split=0.1)
        
        # Evaluate final model
        final_metrics = final_forecaster.evaluate(data_values[:train_size + best_params['sequence_length']], test_data)
        
        # Calculate forecast horizon (sampai akhir 2026)
        data_end_date = df.index[-1]
        forecast_end_date = data_end_date + pd.DateOffset(years=1)
        forecast_days = (forecast_end_date - data_end_date).days
        
        print(f"Forecast horizon: {forecast_days} days")
        
        # Generate forecast
        print(f"Generating LSTM forecast for {target_column}...")
        try:
            forecast = final_forecaster.predict(data_values, steps=forecast_days)
            
            if forecast is None or len(forecast) == 0:
                raise ValueError("Forecast result is empty")
            
            forecast = np.array(forecast).flatten()
            
            if np.isnan(forecast).any() or np.isinf(forecast).any():
                raise ValueError("Forecast contains NaN or infinite values")
            
            # Post-process forecast
            forecast = post_process_forecast(forecast, target_column)
            
            print(f"✓ {target_column} LSTM forecast completed")
            print(f"  Forecast range: {forecast.min():.3f} to {forecast.max():.3f}")
            
        except Exception as e:
            raise ValueError(f"LSTM forecast generation failed: {str(e)}")
        
        # Prepare forecast documents
        forecast_docs = []
        
        try:
            for i in range(len(forecast)):
                forecast_date = df.index[-1] + pd.Timedelta(days=i + 1)
                forecast_date_only = datetime.strptime(forecast_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                
                forecast_value = float(forecast[i])
                if np.isnan(forecast_value) or np.isinf(forecast_value):
                    print(f"Warning: Skipping invalid forecast value at index {i}")
                    continue

                doc = {
                    "forecast_date": forecast_date_only,
                    "timestamp": datetime.now().isoformat(),
                    "source_collection": collection_name,
                    "config_id": config_id,
                    "parameters": {
                        target_column: {
                            "forecast_value": forecast_value,
                            "model_metadata": {
                                "model_type": "LSTM",
                                "sequence_length": best_params["sequence_length"],
                                "batch_size": best_params["batch_size"],
                                "epochs": best_params["epochs"],
                                "train_size": train_size,
                                "total_data_points": len(data_values),
                                "architecture": "128->64->1 (like notebook)"
                            }
                        }
                    }
                }

                if append_column_id:
                    doc["column_id"] = f"{collection_name}_{target_column}"

                forecast_docs.append(doc)
            
        except Exception as e:
            raise ValueError(f"Error preparing forecast documents: {str(e)}")
        
        # Upsert ke collection
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
            "forecast_days": len(forecast_docs),
            "documents_processed": upsert_count,
            "save_collection": save_collection,
            "model_params": best_params,
            "error_metrics": final_metrics if final_metrics else optimization_metrics,
            "forecast_range": {
                "min": float(forecast.min()),
                "max": float(forecast.max())
            },
            "model_type": "LSTM"
        }
        
        print(f"✓ LSTM Analysis completed for {collection_name}.{target_column}")
        return result_summary
        
    except Exception as e:
        print(f"❌ Error in LSTM analysis: {str(e)}")
        raise e
    
    finally:
        if should_close_client:
            client.close()

if __name__ == "__main__":
    # Test LSTM analysis
    run_lstm_analysis(
        collection_name="bmkg-data",
        target_column="RR"
    )
