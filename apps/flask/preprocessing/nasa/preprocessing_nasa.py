from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
import logging 
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NasaPreprocessingError(Exception):
    """Custom exception for NASA POWER preprocessing errors"""
    pass

class NasaDataValidator:
    """Validated NASA POWER data before preprocessing"""
    
    def validate_dataset(self, db, collection_name: str) -> Dict[str, Any]:
        """
        Validates that the dataset contains required columns and is suitable for preprocessing
        Returns a dictionary with validation results
        """
        try:
            # Check if collection exists
            if collection_name not in db.list_collection_names():
                return{
                    'valid': False,
                    'errors': [f"Collection {collection_name} does not exist"]
                }
            # GET sample document to check schema
            sample=db[collection_name].find_one()
            if not sample:
                return{
                    'valid': False,
                    'errors': [f"Collection {collection_name} is empty"]
                }
            # Define required fields for NASA POWER dataset
            required_fields = [
                'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 
                'WS10M', 'WS10M_MAX', 'WD10M'
            ]
            # Check if all required fields are exists
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing required fields: {', '.join(missing_fields)}"]
                }
            # Check if date field exists
            date_fields = ['Date', 'Year', 'month', 'day']
            missing_date_fields = [field for field in date_fields if field not in sample]
            
            if missing_date_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing date fields: {', '.join(missing_date_fields)}"]
                }
            # All validations passed
            return {
                'valid': True,
                'message': "NASA POWER dataset is valid for preprocessing!"
            }
        except Exception as e:
            logger.error(f"Error during dataset validation: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
            
class NasaDataLoader:
    """Loads NASA POWER data from MongoDB into pandas DataFrame"""
    
    def load_data(self, db, collection_name: str) -> pd.DataFrame:
        """Load all data from MongoDB into pd.DataFrame"""
        try:
            # Load all records from MongoDB collection, no pagination
            cursor = db[collection_name].find({})
            df = pd.DataFrame(list(cursor))
            
            if len(df) == 0:
                raise NasaPreprocessingError(f"No data found in collection '{collection_name}'")
            
            # Convert ObjectId into string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            # Ensure date columns is datetime type
            if 'Date' in df.columns and not pd.api.types.is_datetime64_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.warning(f"Failed to convert 'Date' column to datetime: {str(e)}")
            
            # Sort by Date chronologically order
            if 'Date' in df.columns:
                df = df.sort_values('Date')
            
            logger.info(f"Successfully loaded {len(df)} records from '{collection_name}'")
            return df
        except Exception as e:
            error_msg = f"Error loading data from collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
        
class NasaDataSaver:
    """Saves preprocessed NASA POWER data back to new collection_name MongoDB"""
    
    def save_preprocessed_data(
        self,
        db,
        preprocessed_data: pd.DataFrame,
        original_collection_name: str,
    ) -> Dict[str, Any]:
        """Save preprocessed to a new collection, update dataset-meta"""
        try:
            # Generate cleaned collection name
            cleaned_collection_name = f"{original_collection_name}_cleaned"
            
            # Drop the existing cleaned collection if it exists
            if cleaned_collection_name in db.list_collection_names():
                logger.info(f"Dropping existing collection: {cleaned_collection_name}")
                db[cleaned_collection_name].drop()
            
            # Generate new _id values instead of keeping the original ones
            # Create a new DataFrame without the _id column
            if '_id' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('_id', axis=1)
            
            # Convert DataFrame records for MongoDB insertion
            # MongoDB will generate new _id values automatically
            records = preprocessed_data.to_dict('records')
            if records:
                db[cleaned_collection_name].insert_many(records)
                logger.info(f"Inserted {len(records)} records into '{cleaned_collection_name}'")
            
            # Update dataset-meta collection 
            meta_info = self._update_dataset_metadata(
                db,
                original_collection_name,
                cleaned_collection_name,
                len(records)
            )
            return {
                "preprocessedCollections": [cleaned_collection_name],
                "recordsInserted": {
                    "original": db[original_collection_name].count_documents({}),
                    "cleaned": len(records)
                },
                "metadata": meta_info
            }
        except Exception as e:
            error_msg = f"Error saving preprocessed data: {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
        
    def _update_dataset_metadata(
        self,
        db,
        original_collection_name: str,
        cleaned_collection_name: str,
        record_count: int
    ) -> Dict[str, Any]:
        """Only update status and totalRecords in dataset-meta in original collection"""
        try:
            # Find the metadata document for the original collection
            meta_collection = None
            if "dataset_meta" in db.list_collection_names():
                meta_collection = "dataset_meta"
            else:
                logger.warning("No metadata collection found!")
                return {"status": "no_meta_collection"}
            
            # Update only the status, totalRecords, and lastUpdated fields
            result = db[meta_collection].update_one(
                {"collectionName": original_collection_name},
                {"$set": {
                    "status": "preprocessed",
                    "total_records": record_count,
                    "lastUpdated": datetime.now()
                }}
            )
            logger.info(f"Updated metadata for collection '{original_collection_name}'")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {"status": "error", "error": str(e)}
class NasaPreprocessor:
    """Main class for preprocessing NASA POWER datasets"""
    
    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.validator = NasaDataValidator()
        self.loader = NasaDataLoader()
        self.saver = NasaDataSaver()
        self.preprocessing_report = {
            "missing_data": {},
            "outliers": {},
            "smoothing": {},
            "gaps": {},
            "quality_metrics": {},
            "warnings": []
        }
        
    def preprocess(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess NASA POWER dataset
        
        Args:
            options: Dictionary of preprocessing options
            
        Returns:
            Dictionary with preprocessing results and processed dataframe
        """
        try:
            # Default options
            default_options = {
                "smoothing_method": "exponential",  # or "moving_average"
                "window_size": 5,
                "exponential_alpha": 0.2,
                "drop_outliers": True,
                "outlier_methods": ["iqr", "zscore"],
                "iqr_multiplier": 1.5,
                "zscore_threshold": 3,
                "outlier_treatment": "interpolate",  # or "cap" or "remove"
                "fill_missing": True,
                "detect_gaps": True,
                "exclude_tail_data": True,  # Exclude last 5 days (NASA lag)
                "max_gap_interpolate": 90,  # days
                "columns_to_process": [
                    'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 
                    'WS10M', 'WS10M_MAX', 'WD10M'
                ],
                "parameter_configs": {
                    "PRECTOTCORR": {
                        "smoothing_method": None,
                        "apply_outlier_detection": False,
                        "preserve_zeros": True,
                        "validate_range": True,
                        "valid_min": 0.0,
                        "valid_max": 500.0
                    }
                }
            }
            
            # Merge with provided options
            if options is None:
                options = {}
            self.options = {**default_options, **options}
            
            # Validate dataset
            validation_result = self.validator.validate_dataset(self.db, self.collection_name)
            if not validation_result.get('valid', False):
                raise NasaPreprocessingError(f"Validation failed: {validation_result.get('errors', ['Unknown error'])}")
            
            # Load data - this gets all records, not just one page
            df = self.loader.load_data(self.db, self.collection_name)
            logger.info(f"Loaded {len(df)} records from collection '{self.collection_name}'")
            
            # Apply preprocessing - WILL BE IMPLEMENTED IN FUTURE
            processed_df = self._apply_preprocessing(df)
            
            # Save processed data
            save_result = self.saver.save_preprocessed_data(
                self.db, 
                processed_df, 
                self.collection_name
            )
            
            return {
                "status": "success",
                "message": "NASA POWER dataset preprocessed successfully",
                "collection": self.collection_name,
                "preprocessedData": processed_df.head(10).to_dict('records'),
                "recordCount": len(processed_df),
                "originalRecordCount": len(df),
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None,
                "preprocessing_report": self.preprocessing_report
            }
        except Exception as e:
            error_msg = f"Error preprocessing NASA POWER data: {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the data with progress tracking"""
        logger.info("Starting NASA POWER data preprocessing...")
        
        processed_df = df.copy()
        total_steps = 7
        current_step = 0
        
        # Helper function untuk log progress
        def log_progress(stage, message):
            nonlocal current_step
            current_step += 1
            percentage = int((current_step / total_steps) * 100)
            # Format khusus yang akan di-parse oleh SSE handler
            logger.info(f"PROGRESS:{percentage}:{stage}:{message}")
        
        # STEP 1: Replace NASA POWER fill values
        log_progress("fill_values", "Replacing fill values with NaN...")
        processed_df = self._replace_fill_values(processed_df)
        
        # STEP 2: Exclude tail data
        log_progress("tail_data", "Checking for tail data to exclude...")
        if self.options.get("exclude_tail_data", True):
            processed_df = self._exclude_tail_data(processed_df)
        
        # STEP 3: Detect and report gaps
        log_progress("gap_detection", "Detecting gaps in time series...")
        if self.options.get("detect_gaps", True):
            self._detect_gaps(processed_df)
        
        # STEP 4: Handle missing values
        log_progress("imputation", "Imputing missing values...")
        if self.options.get("fill_missing", True):
            processed_df = self._impute_missing_values(processed_df)
        
        # STEP 5: Detect and handle outliers
        log_progress("outliers", "Detecting and handling outliers...")
        if self.options.get("drop_outliers", True):
            processed_df = self._handle_outliers(processed_df)
        
        # STEP 6: Apply smoothing
        log_progress("smoothing", "Applying smoothing methods...")
        processed_df = self._apply_smoothing(processed_df)
        
        # STEP 7: Generate quality metrics
        log_progress("quality_metrics", "Generating quality metrics...")
        self._generate_quality_metrics(df, processed_df)
        
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df
    
    
    def _replace_fill_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NASA POWER fill values (-999, -99, -9999) with NaN"""
        logger.info("Replacing fill values with NaN...")
        
        fill_values = [-999, -99, -9999]
        params = self.options.get("columns_to_process", [])
        
        replaced_count = {}
        for param in params:
            if param in df.columns:
                mask = df[param].isin(fill_values)
                count = mask.sum()
                if count > 0:
                    df.loc[mask, param] = np.nan
                    replaced_count[param] = int(count)
                    
        self.preprocessing_report["missing_data"]["fill_values_replaced"] = replaced_count
        logger.info(f"Replaced fill values: {replaced_count}")
        return df
    
    def _exclude_tail_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude last 5-7 days due to NASA POWER data lag"""
        logger.info("Checking for tail data to exclude...")
        
        if 'Date' not in df.columns:
            logger.warning("No Date column found, skipping tail exclusion")
            return df
        
        # Find latest complete date (all parameters have valid data)
        latest_complete = self._find_latest_complete_date(df)
        
        if latest_complete is None:
            logger.warning("Could not determine latest complete date")
            return df
        
        # Count excluded records
        excluded_mask = df['Date'] > latest_complete
        excluded_count = excluded_mask.sum()
        
        if excluded_count > 0:
            excluded_dates = df[excluded_mask]['Date'].tolist()
            df_complete = df[~excluded_mask].copy()
            
            self.preprocessing_report["missing_data"]["tail_data_excluded"] = {
                "count": int(excluded_count),
                "latest_complete_date": str(latest_complete),
                "excluded_dates": [str(d) for d in excluded_dates[:10]],  # First 10 dates
                "reason": "NASA POWER data lag (~5 days)"
            }
            
            logger.info(f"Excluded {excluded_count} tail records after {latest_complete}")
            return df_complete
        
        return df
    
    def _find_latest_complete_date(self, df: pd.DataFrame) -> Optional[datetime]:
        """Find the last date where all parameters have valid data"""
        params = self.options.get("columns_to_process", [])
        
        # Sort by date descending
        df_sorted = df.sort_values('Date', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            # Check if all parameters are valid (not NaN)
            all_valid = all(
                pd.notna(row.get(param)) for param in params if param in df.columns
            )
            
            if all_valid:
                return row['Date']
        
        return None
    
    def _detect_gaps(self, df: pd.DataFrame) -> None:
        """Detect and classify gaps in the time series"""
        logger.info("Detecting gaps in time series data...")    
        
        if 'Date' not in df.columns:
            logger.warning("No Date column found, skipping gap detection")
            return

        df_sorted = df.sort_values('Date').reset_index(drop=True)
        dates = pd.to_datetime(df_sorted['Date'])
        
        # Find expected date range
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
        missing_dates = date_range.difference(dates)
        
        if len(missing_dates) == 0:
            logger.info("No date gaps found")
            self.preprocessing_report["gaps"] = {
                "total_gaps": 0,
                "gap_details": []
            }
            return
        
        # Group consecutive missing dates into gaps
        gaps = []
        current_gap = [missing_dates[0]]
        
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i-1]).days == 1:
                current_gap.append(missing_dates[i])
            else:
                gaps.append(current_gap)
                current_gap = [missing_dates[i]]
        gaps.append(current_gap)
        
        # Classify gaps
        gap_details = []
        small_gaps = medium_gaps = large_gaps = 0
        
        for gap in gaps:
            duration = len(gap)
            gap_info = {
                "start_date": str(gap[0]),
                "end_date": str(gap[-1]),
                "duration_days": duration
            }
            
            if duration <= 7:
                gap_info["type"] = "small"
                gap_info["imputation_method"] = "linear"
                small_gaps += 1
            elif duration <= 30:
                gap_info["type"] = "medium"
                gap_info["imputation_method"] = "spline"
                medium_gaps += 1
            else:
                gap_info["type"] = "large"
                gap_info["imputation_method"] = "seasonal" if duration <= 90 else "none"
                large_gaps += 1
                
                if duration > 90:
                    self.preprocessing_report["warnings"].append(
                        f"Large gap detected: {gap_info['start_date']} to {gap_info['end_date']} "
                        f"({duration} days) - not interpolated"
                    )
            
            gap_details.append(gap_info)
            
        self.preprocessing_report["gaps"] = {
            "total_gaps": len(gaps),
            "small_gaps": small_gaps,
            "medium_gaps": medium_gaps,
            "large_gaps": large_gaps,
            "gap_details": gap_details[:10] # First 10 gaps for report
        }
        logger.info(f"Detected {len(gaps)} gaps: {small_gaps} small, {medium_gaps} medium, {large_gaps} large")
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using appropriate methods"""
        logger.info("Imputing missing values...")
        
        params = self.options.get("columns_to_process", [])
        imputation_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Get parameter-specific config
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            
            # Special handling for precipitation
            if param == "PRECTOTCORR":
                df, imputed_count = self._impute_precipitation(df, param)
            else:
                df, imputed_count = self._impute_standard_parameter(df, param)
            
            if imputed_count > 0:
                imputation_summary[param] = imputed_count
        
        self.preprocessing_report["missing_data"]["imputed_values"] = imputation_summary
        logger.info(f"Imputation summary: {imputation_summary}")
        
        return df
    
    def _impute_precipitation(self, df: pd.DataFrame, param: str) -> tuple:
        """Special imputation for precipitation data"""
        missing_mask = df[param].isna()
        imputed_count = 0
        
        if not missing_mask.any():
            return df, imputed_count
        
        # For small gaps, use context-aware imputation
        missing_indices = np.where(missing_mask)[0]
        
        for idx in missing_indices:
            # Get surrounding context
            before_idx = max(0, idx - 1)
            after_idx = min(len(df) - 1, idx + 1)
            
            before_val = df.iloc[before_idx][param] if before_idx < idx else np.nan
            after_val = df.iloc[after_idx][param] if after_idx > idx else np.nan
            
            # Context-aware imputation
            if pd.notna(before_val) and pd.notna(after_val):
                if before_val == 0 and after_val == 0:
                    # No rain before and after - assume no rain
                    df.loc[df.index[idx], param] = 0.0
                else:
                    # Linear interpolation
                    df.loc[df.index[idx], param] = (before_val + after_val) / 2
                imputed_count += 1
            elif pd.notna(before_val):
                # Forward fill
                df.loc[df.index[idx], param] = before_val
                imputed_count += 1
            elif pd.notna(after_val):
                # Backward fill
                df.loc[df.index[idx], param] = after_val
                imputed_count += 1
        
        return df, imputed_count
    
    def _impute_standard_parameter(self, df: pd.DataFrame, param: str) -> tuple:
        """Standard imputation for non-precipitation parameters"""
        missing_mask = df[param].isna()
        original_missing_count = missing_mask.sum()
        
        if original_missing_count == 0:
            return df, 0
        
        # Linear interpolation for small gaps
        df[param] = df[param].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline interpolation for remaining gaps
        remaining_missing = df[param].isna().sum()
        if remaining_missing > 0 and remaining_missing < original_missing_count:
            df[param] = df[param].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Forward/backward fill for edge cases
        df[param] = df[param].fillna(method='ffill', limit=3)
        df[param] = df[param].fillna(method='bfill', limit=3)
        
        final_missing = df[param].isna().sum()
        imputed_count = original_missing_count - final_missing
        
        return df, int(imputed_count)
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using IQR and Z-score methods"""
        logger.info("Detecting and handling outliers...")
        
        params = self.options.get("columns_to_process", [])
        outlier_methods = self.options.get("outlier_methods", ["iqr", "zscore"])
        outlier_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Skip outlier detection for precipitation
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            if not param_config.get("apply_outlier_detection", True):
                continue
            
            # Detect outliers using both methods
            outliers_iqr = self._detect_outliers_iqr(df, param) if "iqr" in outlier_methods else pd.Series([False] * len(df))
            outliers_zscore = self._detect_outliers_zscore(df, param) if "zscore" in outlier_methods else pd.Series([False] * len(df))
            
            # Intersection: both methods must agree
            outliers_mask = outliers_iqr & outliers_zscore
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                # Handle outliers
                df = self._treat_outliers(df, param, outliers_mask)
                outlier_summary[param] = int(outlier_count)
        
        self.preprocessing_report["outliers"] = {
            "total_outliers": sum(outlier_summary.values()),
            "by_parameter": outlier_summary,
            "methods_used": outlier_methods,
            "treatment": self.options.get("outlier_treatment", "interpolate")
        }
        
        logger.info(f"Outlier detection summary: {outlier_summary}")
        
        return df
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, param: str) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = df[param].quantile(0.25)
        Q3 = df[param].quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = self.options.get("iqr_multiplier", 1.5)
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (df[param] < lower_bound) | (df[param] > upper_bound)
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, param: str) -> pd.Series:
        """Detect outliers using Z-score method"""
        mean = df[param].mean()
        std = df[param].std()
        
        if std == 0:
            return pd.Series([False] * len(df))
        
        z_scores = np.abs((df[param] - mean) / std)
        threshold = self.options.get("zscore_threshold", 3)
        
        return z_scores > threshold
    
    def _treat_outliers(self, df: pd.DataFrame, param: str, outliers_mask: pd.Series) -> pd.DataFrame:
        """Treat detected outliers"""
        treatment = self.options.get("outlier_treatment", "interpolate")
        
        if treatment == "interpolate":
            # Replace outliers with NaN then interpolate
            df.loc[outliers_mask, param] = np.nan
            df[param] = df[param].interpolate(method='linear', limit=3, limit_direction='both')
        
        elif treatment == "cap":
            # Winsorization - cap at boundary values
            Q1 = df[param].quantile(0.25)
            Q3 = df[param].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.options.get("iqr_multiplier", 1.5)
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df.loc[df[param] < lower_bound, param] = lower_bound
            df.loc[df[param] > upper_bound, param] = upper_bound
        
        elif treatment == "remove":
            # Remove rows with outliers (not recommended)
            df = df[~outliers_mask]
        
        return df
    
    def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to appropriate parameters"""
        logger.info("Applying smoothing methods...")
        
        params = self.options.get("columns_to_process", [])
        smoothing_method = self.options.get("smoothing_method", "exponential")
        smoothing_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Check parameter-specific config
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            param_smoothing = param_config.get("smoothing_method", smoothing_method)
            
            if param_smoothing is None:
                # Skip smoothing for this parameter (e.g., PRECTOTCORR)
                smoothing_summary[param] = "none"
                continue
            
            # Apply smoothing
            if param_smoothing == "moving_average":
                df[param] = self._smooth_moving_average(df[param])
                smoothing_summary[param] = "moving_average"
            elif param_smoothing == "exponential":
                df[param] = self._smooth_exponential(df[param])
                smoothing_summary[param] = "exponential"
        
        self.preprocessing_report["smoothing"] = {
            "method": smoothing_method,
            "parameters_smoothed": smoothing_summary
        }
        
        logger.info(f"Smoothing applied: {smoothing_summary}")
        
        return df
    
    def _smooth_moving_average(self, series: pd.Series) -> pd.Series:
        """Apply moving average smoothing"""
        window_size = self.options.get("window_size", 5)
        return series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    def _smooth_exponential(self, series: pd.Series) -> pd.Series:
        """Apply exponential smoothing"""
        alpha = self.options.get("exponential_alpha", 0.2)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def _generate_quality_metrics(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
        """Generate quality metrics for preprocessing report"""
        logger.info("Generating quality metrics...")
        
        params = self.options.get("columns_to_process", [])
        
        # Calculate completeness
        total_cells = len(processed_df) * len(params)
        missing_cells = sum(processed_df[param].isna().sum() for param in params if param in processed_df.columns)
        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate data modification percentage
        original_count = len(original_df)
        processed_count = len(processed_df)
        records_removed = original_count - processed_count
        
        self.preprocessing_report["quality_metrics"] = {
            "original_records": int(original_count),
            "processed_records": int(processed_count),
            "records_removed": int(records_removed),
            "completeness_percentage": round(completeness, 2),
            "data_quality": "high" if completeness > 95 else "medium" if completeness > 90 else "low"
        }
        
        logger.info(f"Quality metrics: {completeness:.2f}% complete, {records_removed} records removed")
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df