from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "weather_analysis"
COLLECTION_NAME = "weather_data"

class WeatherAnalyzer:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.scaler = StandardScaler()
    
    def get_data_from_mongodb(self, start_date=None, end_date=None):
        """Mengambil data dari MongoDB"""
        try:
            query = {}
            if start_date and end_date:
                query = {
                    "tanggal": {
                        "$gte": datetime.strptime(start_date, "%Y-%m-%d"),
                        "$lte": datetime.strptime(end_date, "%Y-%m-%d")
                    }
                }
            
            cursor = self.collection.find(query).sort("tanggal", 1)
            data = list(cursor)
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            df.set_index('tanggal', inplace=True)
            df.index.freq = 'D'
            
            return df
        except Exception as e:
            print(f"Error getting data from MongoDB: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocessing data sesuai dengan script yang diberikan"""
        if df is None or df.empty:
            return None
        
        # Menghitung jumlah nilai null (8888)
        jumlah_angka_delapan = (df['Curah_Hujan'] == 8888).sum()
        print(f'Jumlah null : {jumlah_angka_delapan}')
        
        # Mengganti nilai spesifik dengan 0
        nilai_spesifik = 8888
        cols_to_replace = ['Curah_Hujan', 'Kelembaban', 'Suhu']
        
        for col in cols_to_replace:
            if col in df.columns:
                df[col] = df[col].replace(nilai_spesifik, 0)
        
        # Mengisi data kosong dengan mean
        cols_to_fill = ['Curah_Hujan', 'Kelembaban', 'Suhu']
        
        for col in cols_to_fill:
            if col in df.columns:
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
        
        # Menangani outliers menggunakan IQR
        if 'Curah_Hujan' in df.columns:
            Q1 = df['Curah_Hujan'].quantile(0.25)
            Q3 = df['Curah_Hujan'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identifikasi outliers
            outliers = df[(df['Curah_Hujan'] < lower_bound) | (df['Curah_Hujan'] > upper_bound)]
            print(f"Jumlah data outlier: {outliers.shape[0]}")
            
            # Replace outliers dengan median
            outlier_mask = (df['Curah_Hujan'] < lower_bound) | (df['Curah_Hujan'] > upper_bound)
            df.loc[outlier_mask, 'Curah_Hujan'] = df['Curah_Hujan'].median()
        
        # Smoothing data dengan rolling window
        if 'Curah_Hujan' in df.columns:
            df['Curah_Hujan_Transformed'] = df['Curah_Hujan'].rolling(window=7, center=True).mean()
            df['Curah_Hujan_Transformed'] = df['Curah_Hujan_Transformed'].fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def holt_winter_analysis(self, df, column='Curah_Hujan_Transformed', periods=30):
        """Melakukan analisis Holt-Winter"""
        try:
            if df is None or column not in df.columns:
                return None
            
            # Pastikan data tidak memiliki nilai negatif atau nol untuk multiplicative
            data_series = df[column].copy()
            
            # Jika ada nilai negatif atau nol, gunakan additive method
            if (data_series <= 0).any():
                trend_method = 'add'
                seasonal_method = 'add'
            else:
                trend_method = 'add'
                seasonal_method = 'mul'
            
            # Fit model Holt-Winter
            model = ExponentialSmoothing(
                data_series,
                trend=trend_method,
                seasonal=seasonal_method,
                seasonal_periods=7  # Weekly seasonality
            )
            
            fitted_model = model.fit()
            
            # Prediksi
            forecast = fitted_model.forecast(periods)
            
            # Confidence intervals (simplified)
            forecast_series = pd.Series(forecast, 
                                      index=pd.date_range(start=df.index[-1] + timedelta(days=1), 
                                                         periods=periods, freq='D'))
            
            return {
                'model_params': {
                    'alpha': fitted_model.params['smoothing_level'],
                    'beta': fitted_model.params['smoothing_trend'] if fitted_model.params['smoothing_trend'] else 0,
                    'gamma': fitted_model.params['smoothing_seasonal'] if fitted_model.params['smoothing_seasonal'] else 0
                },
                'fitted_values': fitted_model.fittedvalues.to_dict(),
                'forecast': forecast_series.to_dict(),
                'aic': fitted_model.aic,
                'mse': np.mean((data_series - fitted_model.fittedvalues) ** 2)
            }
        
        except Exception as e:
            print(f"Error in Holt-Winter analysis: {e}")
            return None

# Initialize analyzer
analyzer = WeatherAnalyzer()

@app.route('/api/weather-data', methods=['GET'])
def get_weather_data():
    """Endpoint untuk mengambil data cuaca"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        df = analyzer.get_data_from_mongodb(start_date, end_date)
        
        if df is None:
            return jsonify({'error': 'No data found'}), 404
        
        # Convert to JSON format
        data = []
        for index, row in df.iterrows():
            data.append({
                'tanggal': index.strftime('%Y-%m-%d'),
                'Curah_Hujan': float(row.get('Curah_Hujan', 0)),
                'Kelembaban': float(row.get('Kelembaban', 0)),
                'Suhu': float(row.get('Suhu', 0))
            })
        
        return jsonify({
            'success': True,
            'data': data,
            'total_records': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holt-winter-analysis', methods=['POST'])
def holt_winter_analysis():
    """Endpoint untuk analisis Holt-Winter"""
    try:
        data = request.json
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        periods = data.get('periods', 30)
        column = data.get('column', 'Curah_Hujan')
        
        # Get and preprocess data
        df = analyzer.get_data_from_mongodb(start_date, end_date)
        if df is None:
            return jsonify({'error': 'No data found'}), 404
        
        # Preprocess data
        df_processed = analyzer.preprocess_data(df)
        if df_processed is None:
            return jsonify({'error': 'Failed to preprocess data'}), 500
        
        # Perform Holt-Winter analysis
        analysis_result = analyzer.holt_winter_analysis(df_processed, 'Curah_Hujan_Transformed', periods)
        
        if analysis_result is None:
            return jsonify({'error': 'Failed to perform Holt-Winter analysis'}), 500
        
        # Prepare response data
        original_data = []
        for index, row in df_processed.iterrows():
            original_data.append({
                'tanggal': index.strftime('%Y-%m-%d'),
                'original': float(row.get('Curah_Hujan', 0)),
                'transformed': float(row.get('Curah_Hujan_Transformed', 0))
            })
        
        forecast_data = []
        for date_str, value in analysis_result['forecast'].items():
            forecast_data.append({
                'tanggal': pd.to_datetime(date_str).strftime('%Y-%m-%d'),
                'forecast': float(value)
            })
        
        fitted_data = []
        for date_str, value in analysis_result['fitted_values'].items():
            fitted_data.append({
                'tanggal': pd.to_datetime(date_str).strftime('%Y-%m-%d'),
                'fitted': float(value)
            })
        
        return jsonify({
            'success': True,
            'analysis': {
                'model_params': analysis_result['model_params'],
                'metrics': {
                    'aic': analysis_result['aic'],
                    'mse': analysis_result['mse']
                },
                'original_data': original_data,
                'fitted_values': fitted_data,
                'forecast': forecast_data
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Endpoint untuk upload data ke MongoDB"""
    try:
        data = request.json
        
        if not data or 'records' not in data:
            return jsonify({'error': 'Invalid data format'}), 400
        
        records = data['records']
        
        # Convert and insert data
        processed_records = []
        for record in records:
            processed_record = {
                'tanggal': datetime.strptime(record['tanggal'], '%Y-%m-%d'),
                'Curah_Hujan': float(record.get('Curah_Hujan', 0)),
                'Kelembaban': float(record.get('Kelembaban', 0)),
                'Suhu': float(record.get('Suhu', 0)),
                'created_at': datetime.now()
            }
            processed_records.append(processed_record)
        
        # Insert to MongoDB
        result = analyzer.collection.insert_many(processed_records)
        
        return jsonify({
            'success': True,
            'message': f'Successfully inserted {len(result.inserted_ids)} records',
            'inserted_count': len(result.inserted_ids)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        analyzer.client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'mongodb': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)