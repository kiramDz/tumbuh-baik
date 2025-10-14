import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from bson import ObjectId
import traceback
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


class DataValidator:
    """Validates dataset before preprocessing"""
    
    @staticmethod
    def validate_dataset(db, collection_name: str) -> Dict[str, Any]:
        """
        Validate dataset structure and quality
        
        Returns:
            dict: Validation result with 'valid', 'errors', 'warnings', 'stats'
        """
        errors = []
        warnings = []
        stats = {}
        
        try:
            # Check if collection exists
            if collection_name not in db.list_collection_names():
                errors.append(f"Collection '{collection_name}' not found")
                return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}
            
            # Get collection stats
            doc_count = db[collection_name].count_documents({})
            stats['total_records'] = doc_count
            
            if doc_count == 0:
                errors.append("Collection is empty")
                return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}
            
            if doc_count < 100:
                warnings.append(f"Low data count ({doc_count}). Results may not be reliable.")
            
            # Get sample document to check structure
            sample = db[collection_name].find_one()
            if not sample:
                errors.append("Could not retrieve sample document")
                return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}
            
            # Check for Date field
            if 'Date' not in sample:
                errors.append("Missing required 'Date' field")
            else:
                # Verify Date is datetime
                if not isinstance(sample['Date'], datetime):
                    errors.append("'Date' field must be datetime type")
            
            # Get all available columns
            available_columns = [k for k in sample.keys() if k not in ['_id', '__v']]
            stats['columns'] = available_columns
            
            # Check for meteorological variables
            expected_vars = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad', 'UWND', 'VWND']
            found_vars = [v for v in expected_vars if v in available_columns]
            stats['meteorological_vars'] = found_vars
            
            if len(found_vars) == 0:
                warnings.append("No standard meteorological variables found")
            
            # Check for missing data
            pipeline = [
                {"$project": {
                    **{var: {"$cond": [{"$eq": [f"${var}", None]}, 1, 0]} 
                       for var in found_vars}
                }},
                {"$group": {
                    "_id": None,
                    **{f"{var}_missing": {"$sum": f"${var}"} for var in found_vars}
                }}
            ]
            
            missing_stats = list(db[collection_name].aggregate(pipeline))
            if missing_stats:
                missing_counts = {k: v for k, v in missing_stats[0].items() if k != '_id'}
                stats['missing_counts'] = missing_counts
                
                # Warn if too many missing values
                for var, count in missing_counts.items():
                    pct = (count / doc_count) * 100
                    if pct > 50:
                        warnings.append(f"{var}: {pct:.1f}% missing values")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}


class MongoDataLoader:
    """Handles loading data from MongoDB into pandas DataFrames"""
    
    @staticmethod
    def load_collection_to_dataframe(db, collection_name: str, 
                                    chunk_size: int = 10000) -> pd.DataFrame:
        """
        Load MongoDB collection into pandas DataFrame with chunking
        
        Args:
            db: MongoDB database instance
            collection_name: Name of collection to load
            chunk_size: Number of documents to load at once
            
        Returns:
            pd.DataFrame: Combined dataframe from all chunks
        """
        logger.info(f"Loading collection '{collection_name}'...")
        
        chunks = []
        skip = 0
        total_loaded = 0
        
        while True:
            # Load chunk
            cursor = db[collection_name].find().skip(skip).limit(chunk_size)
            chunk_data = list(cursor)
            
            if not chunk_data:
                break
            
            # Convert to DataFrame
            df_chunk = pd.DataFrame(chunk_data)
            chunks.append(df_chunk)
            
            total_loaded += len(chunk_data)
            skip += chunk_size
            
            logger.info(f"  Loaded {total_loaded} records...")
        
        if not chunks:
            raise PreprocessingError("No data loaded from collection")
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Remove MongoDB-specific fields
        df = df.drop(columns=['_id', '__v'], errors='ignore')
        
        # Set Date as index if available
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()
        
        logger.info(f"Successfully loaded {len(df)} records")
        
        return df
    
    @staticmethod
    def split_by_variable_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split combined DataFrame into separate DataFrames by variable type
        Similar to the original preprocessing that had separate files
        
        Args:
            df: Combined DataFrame with all variables
            
        Returns:
            dict: Dictionary with variable types as keys and DataFrames as values
        """
        data_dict = {}
        
        # Define variable groupings
        var_groups = {
            'radiation': ['SWRad', 'SWRad_StDev', 'SWRad_Max'],
            'rainfall': ['Prec', 'Prec_StDev', 'Prec_Time'],
            'humidity': ['RH'],
            'sst': ['SST'],
            'wind': ['WSPD', 'WDIR', 'UWND', 'VWND'],
            'temperature': [col for col in df.columns if col.startswith('TEMP_')]
        }
        
        for var_type, var_list in var_groups.items():
            available_vars = [v for v in var_list if v in df.columns]
            if available_vars:
                data_dict[var_type] = df[available_vars].copy()
                logger.info(f"  {var_type}: {len(available_vars)} variables found")
        
        return data_dict


class QualityFilter:
    """Handles quality filtering and data cleaning"""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, variable_type: str) -> pd.DataFrame:
        """
        Handle missing values based on variable type
        
        Args:
            df: DataFrame to process
            variable_type: Type of variable (radiation, rainfall, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        if df is None or df.empty:
            return df
        
        df_imputed = df.copy()
        variables = [col for col in df_imputed.columns 
                    if pd.api.types.is_numeric_dtype(df_imputed[col])]
        
        logger.info(f"Handling missing values for {variable_type}: {variables}")
        
        for var in variables:
            missing_count = df_imputed[var].isna().sum()
            if missing_count == 0:
                continue
            
            logger.info(f"  {var}: {missing_count} missing values")
            
            if variable_type == 'radiation':
                # Time-based interpolation for radiation
                df_imputed[var] = df_imputed[var].interpolate(method='time', limit=3)
                remaining = df_imputed[var].isna().sum()
                if remaining > 0:
                    df_imputed[var] = df_imputed[var].fillna(method='ffill', limit=2)
                
            elif variable_type == 'rainfall':
                # For precipitation, missing often means no rain
                if var == 'Prec':
                    df_imputed[var] = df_imputed[var].fillna(0)
                else:
                    df_imputed[var] = df_imputed[var].interpolate(method='linear', limit=2)
                
            elif variable_type == 'humidity':
                # Forward fill then backward fill for humidity
                df_imputed[var] = df_imputed[var].fillna(method='ffill', limit=2)
                remaining = df_imputed[var].isna().sum()
                if remaining > 0:
                    df_imputed[var] = df_imputed[var].fillna(method='bfill', limit=2)
                
            elif variable_type in ['sst', 'temperature']:
                # Linear interpolation for temperature
                df_imputed[var] = df_imputed[var].interpolate(
                    method='linear', limit_direction='both', limit=3
                )
                
            elif variable_type == 'wind':
                # Linear interpolation for wind
                df_imputed[var] = df_imputed[var].interpolate(
                    method='linear', limit_direction='both', limit=2
                )
            
            final_missing = df_imputed[var].isna().sum()
            filled = missing_count - final_missing
            logger.info(f"    Filled {filled}/{missing_count} values. {final_missing} remain missing")
        
        return df_imputed
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, variable: str, 
                       method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: DataFrame to process
            variable: Variable name to check for outliers
            method: Method to use ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers marked/replaced
        """
        df_result = df.copy()
        
        if variable not in df_result.columns:
            return df_result
        
        if not pd.api.types.is_numeric_dtype(df_result[variable]):
            logger.warning(f"Skipping outlier detection for non-numeric: {variable}")
            return df_result
        
        valid_data = df_result[variable].dropna()
        if len(valid_data) == 0:
            return df_result
        
        # Check for too many NaN values
        nan_pct = (df_result[variable].isna().sum() / len(df_result)) * 100
        if nan_pct > 50:
            logger.warning(f"Skipping {variable}: {nan_pct:.1f}% NaN values")
            return df_result
        
        try:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(valid_data))
                outliers = z_scores > threshold
                outlier_indices = valid_data.index[outliers]
                
            elif method == 'iqr':
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_indices = valid_data[
                    (valid_data < lower_bound) | (valid_data > upper_bound)
                ].index
            else:
                logger.warning(f"Unknown outlier method: {method}")
                return df_result
            
            if len(outlier_indices) > 0:
                outlier_pct = (len(outlier_indices) / len(valid_data)) * 100
                logger.info(f"Detected {len(outlier_indices)} outliers in {variable} ({outlier_pct:.2f}%)")
                
                # Mark outliers
                outlier_col = f"is_outlier_{variable}"
                df_result[outlier_col] = False
                df_result.loc[outlier_indices, outlier_col] = True
                
                # Replace with NaN
                df_result.loc[outlier_indices, variable] = np.nan
            else:
                logger.info(f"No outliers detected in {variable}")
                
        except Exception as e:
            logger.error(f"Error detecting outliers for {variable}: {str(e)}")
        
        return df_result
    
    @staticmethod
    def validate_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate physical ranges for meteorological variables
        
        Args:
            df: DataFrame to validate
            
        Returns:
            tuple: (cleaned DataFrame, list of warnings)
        """
        warnings = []
        df_clean = df.copy()
        
        # Define physical ranges
        ranges = {
            'SST': (-2, 35),      # Sea Surface Temperature in °C
            'RH': (0, 100),       # Relative Humidity in %
            'Prec': (0, 500),     # Precipitation in mm (daily max ~500mm)
            'WSPD': (0, 50),      # Wind Speed in m/s (max ~50 m/s)
            'SWRad': (0, 1400),   # Solar Radiation in W/m² (max ~1400)
            'UWND': (-50, 50),    # Zonal Wind in m/s
            'VWND': (-50, 50),    # Meridional Wind in m/s
            'WDIR': (0, 360)      # Wind Direction in degrees
        }
        
        for var, (min_val, max_val) in ranges.items():
            if var in df_clean.columns:
                # Count out-of-range values
                out_of_range = (
                    (df_clean[var] < min_val) | 
                    (df_clean[var] > max_val)
                ).sum()
                
                if out_of_range > 0:
                    pct = (out_of_range / len(df_clean)) * 100
                    warnings.append(
                        f"{var}: {out_of_range} values ({pct:.2f}%) outside "
                        f"physical range [{min_val}, {max_val}]"
                    )
                    logger.warning(warnings[-1])
                    
                    # Set out-of-range values to NaN
                    df_clean.loc[
                        (df_clean[var] < min_val) | (df_clean[var] > max_val),
                        var
                    ] = np.nan
        
        return df_clean, warnings


class BuoyPreprocessor:
    """Main preprocessing class for buoy data"""
    
    def __init__(self, db, collection_name: str, location_code: Optional[str] = None):
        """
        Initialize preprocessor
        
        Args:
            db: MongoDB database instance
            collection_name: Name of collection to process
            location_code: Optional location identifier (e.g., '0N90E')
        """
        self.db = db
        self.collection_name = collection_name
        self.location_code = location_code or self._extract_location_code()
        self.data_loader = MongoDataLoader()
        self.quality_filter = QualityFilter()
        self.validator = DataValidator()
        
    def _extract_location_code(self) -> str:
        """Extract location code from collection name"""
        # Example: buoys_location_0n90e_ -> 0N90E
        parts = self.collection_name.split('_')
        if len(parts) >= 3:
            location = parts[-1] if parts[-1] else parts[-2]
            return location.upper().replace('_', '')
        return 'UNKNOWN'
    
    def preprocess(self, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main preprocessing pipeline
        
        Args:
            options: Preprocessing options/configuration
            
        Returns:
            dict: Preprocessing results with cleaned data and metadata
        """
        if options is None:
            options = {}
        
        # Default options
        default_options = {
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0,
            'handle_negative_precip': True,
            'validate_ranges': True,
            'create_combined': True
        }
        options = {**default_options, **options}
        
        logger.info(f"Starting preprocessing for {self.collection_name}")
        logger.info(f"Location: {self.location_code}")
        logger.info(f"Options: {options}")
        
        results = {
            'location_code': self.location_code,
            'collection_name': self.collection_name,
            'options': options,
            'cleaned_data': {},
            'statistics': {},
            'warnings': [],
            'processing_steps': []
        }
        
        try:
            # Step 1: Load data
            self._add_step(results, "Loading data from MongoDB")
            df_raw = self.data_loader.load_collection_to_dataframe(
                self.db, self.collection_name
            )
            results['statistics']['raw_records'] = len(df_raw)
            results['statistics']['date_range'] = {
                'start': df_raw.index.min().isoformat(),
                'end': df_raw.index.max().isoformat()
            }
            
            # Step 2: Split by variable type
            self._add_step(results, "Splitting data by variable types")
            data_dict = self.data_loader.split_by_variable_type(df_raw)
            results['statistics']['variable_types'] = list(data_dict.keys())
            
            # Step 3: Process each variable type
            for var_type, df in data_dict.items():
                self._add_step(results, f"Processing {var_type} data")
                
                # Handle missing values
                df_clean = self.quality_filter.handle_missing_values(df, var_type)
                
                # Handle outliers for each variable
                for col in df_clean.columns:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean = self.quality_filter.handle_outliers(
                            df_clean, col,
                            method=options['outlier_method'],
                            threshold=options['outlier_threshold']
                        )
                
                # Validate physical ranges
                if options['validate_ranges']:
                    df_clean, warnings = self.quality_filter.validate_ranges(df_clean)
                    results['warnings'].extend(warnings)
                
                # Store cleaned data
                results['cleaned_data'][var_type] = df_clean
                
                # Calculate statistics
                results['statistics'][var_type] = {
                    'variables': list(df_clean.columns),
                    'records': len(df_clean),
                    'missing_pct': {
                        col: (df_clean[col].isna().sum() / len(df_clean) * 100)
                        for col in df_clean.columns
                        if pd.api.types.is_numeric_dtype(df_clean[col])
                    }
                }
            
            # Step 4: Handle negative precipitation
            if options['handle_negative_precip'] and 'rainfall' in results['cleaned_data']:
                self._add_step(results, "Handling negative precipitation")
                self._handle_negative_precipitation(results)
            
            # Step 5: Create combined dataset
            if options['create_combined']:
                self._add_step(results, "Creating combined dataset")
                combined_df = self._create_combined_dataset(results['cleaned_data'])
                results['cleaned_data']['combined'] = combined_df
                
                # Calculate correlation matrix
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    results['statistics']['correlations'] = (
                        combined_df[numeric_cols].corr().to_dict()
                    )
            
            self._add_step(results, "Preprocessing completed successfully")
            logger.info("Preprocessing completed successfully")
            
            return results
            
        except Exception as e:
            error_msg = f"Preprocessing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PreprocessingError(error_msg)
    
    def _handle_negative_precipitation(self, results: Dict):
        """Handle negative precipitation values"""
        if 'rainfall' not in results['cleaned_data']:
            return
        
        df_rain = results['cleaned_data']['rainfall']
        if 'Prec' not in df_rain.columns:
            return
        
        neg_count = (df_rain['Prec'] < 0).sum()
        if neg_count == 0:
            logger.info("No negative precipitation values found")
            return
        
        logger.info(f"Found {neg_count} negative precipitation values")
        
        # Simple strategy: set to trace amount (0.01 mm)
        # In production, could use ML model as in original code
        df_rain.loc[df_rain['Prec'] < 0, 'Prec'] = 0.01
        
        results['cleaned_data']['rainfall'] = df_rain
        results['warnings'].append(
            f"Set {neg_count} negative precipitation values to 0.01 mm (trace)"
        )
    
    def _create_combined_dataset(self, cleaned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create combined dataset from all variable types"""
        key_vars = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad', 'UWND', 'VWND']
        
        # Collect all available variables
        dfs_to_merge = []
        for var_type, df in cleaned_data.items():
            for var in key_vars:
                if var in df.columns:
                    dfs_to_merge.append(df[[var]])
        
        if not dfs_to_merge:
            logger.warning("No key variables found for combined dataset")
            return pd.DataFrame()
        
        # Merge all DataFrames on index (Date)
        combined = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            combined = combined.join(df, how='outer')
        
        # Add date components
        combined['year'] = combined.index.year
        combined['month'] = combined.index.month
        combined['day'] = combined.index.day
        
        logger.info(f"Created combined dataset with {len(combined)} records and {len(combined.columns)} columns")
        
        return combined
    
    def _add_step(self, results: Dict, step_name: str):
        """Add processing step to results"""
        results['processing_steps'].append({
            'name': step_name,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Step: {step_name}")


class MongoDataSaver:
    """Handles saving processed data back to MongoDB"""
    
    @staticmethod
    def save_processed_data(db, results: Dict[str, Any], 
                        source_collection: str) -> Dict[str, Any]:
        """
        Save processed data back to MongoDB
        
        Args:
            db: MongoDB database instance
            results: Preprocessing results including cleaned data
            source_collection: Source collection name
            
        Returns:
            dict: Metadata about saved collections
        """
        if not results or 'cleaned_data' not in results:
            raise PreprocessingError("No cleaned data to save")
        
        location_code = results.get('location_code', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        processed_collections = []
        record_counts = {}
        
        # Save each variable type to its own collection
        for var_type, df in results['cleaned_data'].items():
            if df is None or df.empty:
                continue
                
            # Create collection name for processed data
            processed_collection = f"buoys_processed_{var_type}_{location_code}_{timestamp}"
            
            # Convert DataFrame to list of dictionaries
            records = df.reset_index().to_dict('records')
            
            # Insert into MongoDB
            if records:
                # Convert all dates to MongoDB datetime
                for record in records:
                    if 'Date' in record and isinstance(record['Date'], (str, pd.Timestamp)):
                        record['Date'] = pd.to_datetime(record['Date']).to_pydatetime()
                        
                db[processed_collection].insert_many(records)
                processed_collections.append(processed_collection)
                record_counts[var_type] = len(records)
                logger.info(f"Saved {len(records)} records to {processed_collection}")
        
        # Create combined dataset if it exists
        if 'combined' in results:
            combined_collection = f"buoys_processed_combined_{location_code}_{timestamp}"
            combined_df = results['combined']
            
            if not combined_df.empty:
                combined_records = combined_df.reset_index().to_dict('records')
                
                # Convert all dates
                for record in combined_records:
                    if 'Date' in record and isinstance(record['Date'], (str, pd.Timestamp)):
                        record['Date'] = pd.to_datetime(record['Date']).to_pydatetime()
                
                db[combined_collection].insert_many(combined_records)
                processed_collections.append(combined_collection)
                record_counts['combined'] = len(combined_records)
                logger.info(f"Saved {len(combined_records)} records to {combined_collection}")
        
        # Save metadata about this processing job
        metadata = {
            "sourceCollection": source_collection,
            "locationCode": location_code,
            "processedCollections": processed_collections,
            "recordCounts": record_counts,
            "preprocessingOptions": results.get('options', {}),
            "preprocessingStats": results.get('statistics', {}),
            "warnings": results.get('warnings', []),
            "processingSteps": results.get('processing_steps', []),
            "processedAt": datetime.now(),
            "status": "preprocessed"
        }
        
        # Save metadata
        metadata_id = db["preprocessing_metadata"].insert_one(metadata).inserted_id
        metadata["_id"] = str(metadata_id)
        
        return metadata


# Export main functions for use in Flask routes
__all__ = [
    'DataValidator',
    'MongoDataLoader',
    'QualityFilter',
    'BuoyPreprocessor',
    'MongoDataSaver',
    'PreprocessingError'
]