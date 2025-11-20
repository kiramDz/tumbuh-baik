from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
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
                
            # Remove any existing __v column
            if '__v' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('__v', axis=1)
                logger.info("Dropped '__v' column from preprocessed data")
            
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
        """Update dataset-meta after preprocessing - Point to cleaned collection"""
        try:
            # Find the metadata collection
            meta_collection = None
            if "dataset_meta" in db.list_collection_names():
                meta_collection = "dataset_meta"
            elif "DatasetMeta" in db.list_collection_names():
                meta_collection = "DatasetMeta"
            else:
                logger.warning("No metadata collection found!")
                return {"status": "no_meta_collection"}
            
            # Get original metadata to preserve other fields
            original_meta = db[meta_collection].find_one({"collectionName": original_collection_name})
            
            if not original_meta:
                logger.warning(f"No metadata found for collection '{original_collection_name}'")
                return {"status": "no_original_metadata"}
            
            # Get cleaned collection columns
            sample_doc = db[cleaned_collection_name].find_one()
            if sample_doc:
                all_columns = list(sample_doc.keys())
                cleaned_columns = [
                    col for col in all_columns
                    if col not in ['_id', '__v'] and not col.startswith('_')
                ]
            else:
                cleaned_columns = []
            
            # cleaned_columns = list(sample_doc.keys()) if sample_doc else []
            
            # Update metadata to point to cleaned collection
            update_fields = {
                "status": "preprocessed",
                "collectionName": cleaned_collection_name,  # ✅ Point to cleaned collection
                "totalRecords": record_count,               # ✅ Correct field name
                "columns": cleaned_columns,                 # ✅ Updated columns
                "lastUpdated": datetime.now(),
                "name": f"{original_meta.get('name', original_collection_name)} (Cleaned)"
            }
            
            # Only include apiConfig if it's an API dataset
            if original_meta.get('isAPI', False):
                update_fields["apiConfig"] = original_meta.get('apiConfig', {})
            
            result = db[meta_collection].update_one(
                {"collectionName": original_collection_name},
                {"$set": update_fields}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated metadata to point to cleaned collection: {cleaned_collection_name}")
            else:
                logger.warning("No metadata was updated")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {"status": "error", "error": str(e)}
        
    # def _update_dataset_metadata(
    #     self,
    #     db,
    #     original_collection_name: str,
    #     cleaned_collection_name: str,
    #     record_count: int
    # ) -> Dict[str, Any]:
    #     """Only update status and totalRecords in dataset-meta in original collection"""
    #     try:
    #         # Find the metadata document for the original collection
    #         meta_collection = None
    #         if "dataset_meta" in db.list_collection_names():
    #             meta_collection = "dataset_meta"
    #         else:
    #             logger.warning("No metadata collection found!")
    #             return {"status": "no_meta_collection"}
            
    #         # Update only the status, totalRecords, and lastUpdated fields
    #         result = db[meta_collection].update_one(
    #             {"collectionName": original_collection_name},
    #             {"$set": {
    #                 "status": "preprocessed",
    #                 "total_records": record_count,
    #                 "lastUpdated": datetime.now()
    #             }}
    #         )
    #         logger.info(f"Updated metadata for collection '{original_collection_name}'")
    #         return {"status": "success"}
    #     except Exception as e:
    #         logger.error(f"Error updating metadata: {str(e)}")
    #         return {"status": "error", "error": str(e)}
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
            "r2_validation": {},
            "model_coverage": {},
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
                "calculate_r2": True,
                "r2_threshold": 0.85,
                "calculate_coverage": True,
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
        total_steps = 9
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
        
        # STEP 7: Calculate R² validation
        log_progress("r2_validation", "Calculating preprocessing R² validation...")
        if self.options.get("calculate_r2", True):
            self._calculate_preprocessing_r2(df, processed_df)
            
        # STEP 8: Calculate model coverage 
        log_progress("model_coverage", "Calculating model coverage analysis...")
        if self.options.get("calculate_coverage", True):
            self._calculate_model_coverage(processed_df)
        
        # STEP 9: Generate quality metrics
        log_progress("quality_metrics", "Generating quality metrics...")
        self._generate_quality_metrics(df, processed_df)
        
        # log_progress("quality_metrics", "Generating quality metrics...")
        # self._generate_quality_metrics(df, processed_df)
        
        # numeric_columns = [col for col in processed_df.select_dtypes(include=[np.number]).columns]
        
        # for col in numeric_columns:
        #     if col in processed_df.columns:
        #         processed_df[col] = processed_df[col].round(2)
        
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
    
    def _calculate_preprocessing_r2(
        self, 
        original_df: pd.DataFrame,
        preprocessed_df: pd.DataFrame
    ) -> None:
        """
        Measure how well preprocessed data preserves original data trends using R² score
        """
        logger.info("Calculating preprocessing R² validation...")
        
        params = self.options.get("columns_to_process", [])
        r2_threshold = self.options.get("r2_threshold", 0.85)
        r2_results = {}
        
        for param in params: 
            if param not in original_df.columns or param not in preprocessed_df.columns:
                continue
            
            try:
                # Get overlapping time periods only (exclude tail data and removed outliers)
                # Find common indices between original and preprocessed
                common_indices = original_df.index.intersection(preprocessed_df.index)
                
                if len (common_indices) == 0:
                    logger.warning(f"No common indices found for parameter {param}")
                    r2_results[param] = {"r2_score": None, "status": "no_common_data"}
                    continue
                
                # Extract overlapping data
                original_values = original_df.loc[common_indices, param]
                preprocessed_values = preprocessed_df.loc[common_indices, param]
                
                # Remove NaN values from both series (must be paired)
                mask = ~(original_values.isna() | preprocessed_values.isna())
                
                if mask.sum() < 10:
                    logger.warning(f"Insufficient valid data points for {param}: {mask.sum()}")
                    r2_results[param] = {"r2_score": None, "status": "insufficient_data"}
                    continue
                
                original_clean = original_values[mask]
                preprocessed_clean = preprocessed_values[mask]
                
                # Calculate R² score
                if len(original_clean) > 0 and original_clean.std() > 0:
                    r2 = r2_score(original_clean, preprocessed_clean)
                    
                    # Determine quality status
                    if r2 >= r2_threshold:
                        status = "excellent"
                    elif r2 >= 0.75:
                        status = "good"
                    elif r2 >= 0.50:
                        status = "fair"
                    else:
                        status = "poor"
                        
                    r2_results[param] = {
                        "r2_score": round(float(r2), 4),
                        "status": status,
                        "data_points": int(len(original_clean)),
                        "original_range": f"{original_clean.min():.3f} to {original_clean.max():.3f}",
                        "preprocessed_range": f"{preprocessed_clean.min():.3f} to {preprocessed_clean.max():.3f}"
                    }
                    
                    if r2 < r2_threshold:
                        self.preprocessing_report["warnings"].append(
                            f"Parameter {param} has low R² score ({r2:.3f} < {r2_threshold})"
                        )
                else:
                    r2_results[param] = {"r2_score": None, "status": "no_variance"}

                
            except Exception as e:
                logger.error(f"Error calculating R² for parameter {param}: {str(e)}")
                r2_results[param] = {"r2_score": None, "status": f"error: {str(e)}"}
        
        # Calculate overall preprocessing quality
        valid_r2_scores = [
            result["r2_score"] for result in r2_results.values() 
            if result["r2_score"] is not None
        ]
        
        if valid_r2_scores:
            overall_r2 = sum(valid_r2_scores) / len(valid_r2_scores)
            parameters_above_threshold = sum(1 for r2 in valid_r2_scores if r2 >= r2_threshold)
            
            overall_quality = {
                "overall_r2": round(float(overall_r2), 4),
                "parameters_above_threshold": parameters_above_threshold,
                "total_parameters": len(valid_r2_scores),
                "threshold": r2_threshold,
                "quality_percentage": round((parameters_above_threshold / len(valid_r2_scores)) * 100, 2)
            }
            
            # Determine overall preprocessing quality
            if overall_r2 >= r2_threshold and parameters_above_threshold >= len(valid_r2_scores) * 0.8:
                overall_quality["preprocessing_quality"] = "excellent"
            elif overall_r2 >= 0.75 and parameters_above_threshold >= len(valid_r2_scores) * 0.6:
                overall_quality["preprocessing_quality"] = "good"
            elif overall_r2 >= 0.50:
                overall_quality["preprocessing_quality"] = "fair"
            else:
                overall_quality["preprocessing_quality"] = "poor"
                
        else:
            overall_quality = {
                "overall_r2": None,
                "preprocessing_quality": "unknown",
                "error": "No valid R² scores calculated"
            }
        
        # Store R² validation results in preprocessing report
        self.preprocessing_report["r2_validation"] = {
            "threshold": r2_threshold,
            "per_parameter": r2_results,
            "overall": overall_quality
        }
        
        logger.info(f"R² validation completed - Overall R²: {overall_quality.get('overall_r2', 'N/A')}")
    
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
    
    def _calculate_model_coverage(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Calculate model applicability coverage for Holt-Winters and LSTM models
        Analyzes data segments that are suitable/unsuitable for each model type
        """
        logger.info("Calculating model coverage analysis...")
        
        params = self.options.get("columns_to_process", [])
        coverage_results = {
            "holt_winters": {"coverage_percentage": 0, "uncovered_breakdown": {}},
            "lstm": {"coverage_percentage": 0, "uncovered_breakdown": {}},
            "per_parameter": {}
        }
        
        total_data_points = len(df)
        
        for param in params:
            if param not in df.columns:
                continue
            
            try:
                param_coverage = self._analyze_parameter_coverage(df, param, total_data_points)
                coverage_results["per_parameter"][param] = param_coverage
            except Exception as e:
                logger.error(f"Error calculating coverage for parameter {param}: {str(e)}")
                coverage_results["per_parameter"][param] = {
                    "holt_winters_coverage": 0,
                    "lstm_coverage": 0,
                    "error": str(e)
                }
        
        # Calculate overall coverage percentages
        if coverage_results["per_parameter"]:
            hw_coverage = [
                result.get("holt_winters_coverage", 0)
                for result in coverage_results["per_parameter"].values()
                if isinstance(result, dict) and "holt_winters_coverage" in result
            ]
            lstm_coverage = [
                result.get("lstm_coverage", 0)
                for result in coverage_results["per_parameter"].values()
                if isinstance(result, dict) and "lstm_coverage" in result
            ]
            
            # Overall coverage (average across parameters)
            overall_hw_coverage = sum(hw_coverage) / len(hw_coverage) if hw_coverage else 0
            overall_lstm_coverage = sum(lstm_coverage) / len(lstm_coverage) if lstm_coverage else 0
            
            # Aggregate uncovered breakdown
            hw_uncovered = self._aggregate_uncovered_breakdown(coverage_results["per_parameter"], "holt_winters")
            lstm_uncovered = self._aggregate_uncovered_breakdown(coverage_results["per_parameter"], "lstm")
            
            coverage_results["holt_winters"] = {
                "coverage_percentage": round(overall_hw_coverage, 2),
                "uncovered_breakdown": hw_uncovered,
                "model_suitability": self._determine_model_suitability(overall_hw_coverage)
            }
            
            coverage_results["lstm"] = {
                "coverage_percentage": round(overall_lstm_coverage, 2),
                "uncovered_breakdown": lstm_uncovered,
                "model_suitability": self._determine_model_suitability(overall_lstm_coverage)
            }
        
        # Store coverage results in preprocessing report
        self.preprocessing_report["model_coverage"] = coverage_results
        logger.info(f"Model coverage - Holt Winters: {coverage_results['holt_winters']['coverage_percentage']:.1f}%" )
        logger.info(f"Model coverage - LSTM: {coverage_results['lstm']['coverage_percentage']:.1f}%")
        
    def _analyze_parameter_coverage(
        self,
        df: pd.DataFrame,
        param: str,
        total_points: int
    ) -> Dict[str, Any]:
        """
        Analyze coverage for a specific parameter for both model types
        """
        series = df[param].dropna()
        if len(series) < 30: # Need minimum data for analysis
            return {
                "holt_winters_coverage": 0,
                "lstm_coverage": 0,
                "insufficient_data": True
            }
        
        # Analysis results
        coverage_analysis = {
            "data_points": len(series),
            "missing_ratio": (total_points - len(series)) / total_points
        }
        
        # 1. Analyze large gaps (affects both models)
        large_gaps_impact = self._analyze_large_gaps(df, param)
        
        # 2. Analyze extreme outliers (affects both models)
        extreme_outliers_impact = self._analyze_extreme_outliers(series)
        
        # 3. Analyze seasonality (critical for Holt-Winters)
        seasonality_analysis = self._analyze_seasonality(series)
        
        # 4. Analyze stationarity (affects model selection)
        stationarity_analysis = self._analyze_stationarity(series)
        
        # 5. Analyze extreme precipitation events (for PRECTOTCORR)
        precipitation_analysis = self._analyze_precipitation_extremes(series, param) if param == "PRECTOTCORR" else {}
        
        # Calculate Holt-Winters coverage
        hw_coverage = self._calculate_holt_winters_coverage(
            large_gaps_impact,
            extreme_outliers_impact,
            seasonality_analysis,
            stationarity_analysis,
            coverage_analysis
        )
        
        # Calculate LSTM coverage
        lstm_coverage = self._calculate_lstm_coverage(
            large_gaps_impact,
            extreme_outliers_impact,
            precipitation_analysis,
            stationarity_analysis,
            coverage_analysis
        )
        
        return {
            "holt_winters_coverage": hw_coverage["coverage_percentage"],
            "lstm_coverage": lstm_coverage["coverage_percentage"],
            "holt_winters_uncovered": hw_coverage["uncovered_reasons"],
            "lstm_uncovered": lstm_coverage["uncovered_reasons"],
            "analysis_details": {
                "large_gaps": large_gaps_impact,
                "extreme_outliers": extreme_outliers_impact,
                "seasonality": seasonality_analysis,
                "stationarity": stationarity_analysis,
                "precipitation": precipitation_analysis
            }
        }
        
    def _analyze_large_gaps(self, df: pd.DataFrame, param: str) -> Dict[str, Any]:
        """Analyze impact of large gaps (>90 days)"""
        if 'Date' not in df.columns:
            return {"impact_percentage": 0, "large_gaps_count": 0}
        
        # Get gaps from preprocessing report
        gaps_info = self.preprocessing_report.get("gaps", {})
        large_gaps = [gap for gap in gaps_info.get("gap_details", []) if gap.get("duration_days", 0) > 90]
        
        if not large_gaps:
            return {"impact_percentage": 0, "large_gaps_count": 0}
        
        # Calculate impact
        total_gap_days = sum(gap["duration_days"] for gap in large_gaps)
        total_days = (df['Date'].max() - df['Date'].min()).days + 1
        impact_percentage = (total_gap_days / total_days) * 100
        
        return {
            "impact_percentage": round(impact_percentage, 2),
            "large_gaps_count": len(large_gaps),
            "total_gap_days": total_gap_days
        }
    
    def _analyze_extreme_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze extreme outliers that survived double-detection"""
        if len(series) < 10:
            return {"impact_percentage": 0}
        
        # More stringent outlier detection (3.5 sigma)
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return {"impact_percentage": 0}
        
        z_scores = np.abs((series - mean) / std)
        extreme_outliers = z_scores > 3.5  # More stringent than preprocessing (3.0)
        
        impact_percentage = (extreme_outliers.sum() / len(series)) * 100
        
        return {
            "impact_percentage": round(impact_percentage, 2),
            "extreme_outliers_count": int(extreme_outliers.sum()),
            "max_z_score": round(float(z_scores.max()), 2)
        }

    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns (critical for Holt-Winters)"""
        try:
            if len(series) < 730:  # Need at least 2 years for seasonal analysis
                return {"seasonal_strength": 0, "has_clear_seasonality": False, "insufficient_data": True}
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=365, extrapolate_trend='freq')
            
            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            if seasonal_var + residual_var == 0:
                seasonal_strength = 0
            else:
                seasonal_strength = seasonal_var / (seasonal_var + residual_var)
            
            has_clear_seasonality = seasonal_strength > 0.3  # Threshold for clear seasonality
            
            return {
                "seasonal_strength": round(float(seasonal_strength), 3),
                "has_clear_seasonality": has_clear_seasonality,
                "seasonal_variance": round(float(seasonal_var), 3),
                "residual_variance": round(float(residual_var), 3)
            }
            
        except Exception as e:
            logger.warning(f"Error in seasonal analysis: {str(e)}")
            return {"seasonal_strength": 0, "has_clear_seasonality": False, "error": str(e)}

    def _analyze_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze stationarity using Augmented Dickey-Fuller test"""
        try:
            if len(series) < 50:
                return {"is_stationary": False, "insufficient_data": True}
            
            # Perform ADF test
            adf_result = adfuller(series.dropna())
            
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05 indicates stationarity
            
            return {
                "is_stationary": is_stationary,
                "adf_statistic": round(float(adf_result[0]), 4),
                "p_value": round(float(adf_result[1]), 4),
                "critical_values": {k: round(v, 4) for k, v in adf_result[4].items()}
            }
            
        except Exception as e:
            logger.warning(f"Error in stationarity analysis: {str(e)}")
            return {"is_stationary": False, "error": str(e)}
        
    def _analyze_precipitation_extremes(self, series: pd.Series, param: str) -> Dict[str, Any]:
        """Analyze extreme precipitation events (0 vs 500mm range)"""
        if param != "PRECTOTCORR":
            return {}
        
        zero_ratio = (series == 0).sum() / len(series)
        extreme_high = (series > 100).sum() / len(series)  # >100mm per day is extreme
        
        # Calculate range ratio (wide range affects LSTM)
        data_range = series.max() - series.min()
        range_impact = min(data_range / 500, 1.0)  # Normalize to 0-1
        
        return {
            "zero_precipitation_ratio": round(float(zero_ratio), 3),
            "extreme_precipitation_ratio": round(float(extreme_high), 3),
            "range_impact": round(float(range_impact), 3),
            "max_precipitation": round(float(series.max()), 2)
        }

    def _calculate_holt_winters_coverage(self, large_gaps, extreme_outliers, seasonality, stationarity, coverage_analysis) -> Dict[str, Any]:
        """Calculate Holt-Winters model coverage"""
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # Penalize for large gaps (critical for HW)
        large_gap_penalty = large_gaps.get("impact_percentage", 0)
        base_coverage -= large_gap_penalty
        if large_gap_penalty > 0:
            uncovered_reasons["large_gaps"] = large_gap_penalty
        
        # Penalize for extreme outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        base_coverage -= outlier_penalty
        if outlier_penalty > 0:
            uncovered_reasons["extreme_outliers"] = outlier_penalty
        
        # Penalize for lack of seasonality (critical for HW)
        if not seasonality.get("has_clear_seasonality", False):
            seasonal_penalty = 15.0  # 15% penalty for no clear seasonality
            base_coverage -= seasonal_penalty
            uncovered_reasons["no_clear_seasonality"] = seasonal_penalty
        
        # Minor penalty for non-stationarity
        if not stationarity.get("is_stationary", False):
            stationarity_penalty = 5.0
            base_coverage -= stationarity_penalty
            uncovered_reasons["non_stationary"] = stationarity_penalty
        
        # Penalize for missing data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 10  # 10% penalty per 100% missing
        base_coverage -= missing_penalty
        if missing_penalty > 0:
            uncovered_reasons["missing_data"] = missing_penalty
        
        final_coverage = max(0, base_coverage)  # Ensure non-negative
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }

    def _calculate_lstm_coverage(self, large_gaps, extreme_outliers, precipitation, stationarity, coverage_analysis) -> Dict[str, Any]:
        """Calculate LSTM model coverage"""
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # Penalize for large gaps
        large_gap_penalty = large_gaps.get("impact_percentage", 0) * 0.8  # Slightly less critical than HW
        base_coverage -= large_gap_penalty
        if large_gap_penalty > 0:
            uncovered_reasons["large_gaps"] = large_gap_penalty
        
        # Penalize for extreme outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        base_coverage -= outlier_penalty
        if outlier_penalty > 0:
            uncovered_reasons["extreme_outliers"] = outlier_penalty
        
        # Penalize for extreme precipitation events (if applicable)
        if precipitation:
            precip_penalty = precipitation.get("range_impact", 0) * 10  # Up to 10% penalty
            base_coverage -= precip_penalty
            if precip_penalty > 0:
                uncovered_reasons["precipitation_extremes"] = precip_penalty
        
        # Penalize for missing data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 8  # 8% penalty per 100% missing
        base_coverage -= missing_penalty
        if missing_penalty > 0:
            uncovered_reasons["missing_data"] = missing_penalty
        
        final_coverage = max(0, base_coverage)  # Ensure non-negative
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }

    def _aggregate_uncovered_breakdown(self, per_parameter_results: Dict, model_type: str) -> Dict[str, float]:
        """Aggregate uncovered breakdown across all parameters"""
        
        uncovered_key = f"{model_type}_uncovered"
        all_reasons = {}
        
        for param_result in per_parameter_results.values():
            if isinstance(param_result, dict) and uncovered_key in param_result:
                for reason, percentage in param_result[uncovered_key].items():
                    if reason not in all_reasons:
                        all_reasons[reason] = []
                    all_reasons[reason].append(percentage)
        
        # Average the percentages across parameters
        aggregated = {}
        for reason, percentages in all_reasons.items():
            aggregated[reason] = round(sum(percentages) / len(percentages), 2)
        
        return aggregated

    def _determine_model_suitability(self, coverage_percentage: float) -> str:
        """Determine model suitability based on coverage percentage"""
        if coverage_percentage >= 85:
            return "excellent"
        elif coverage_percentage >= 75:
            return "good"
        elif coverage_percentage >= 60:
            return "fair"
        else:
            return "poor"
    
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