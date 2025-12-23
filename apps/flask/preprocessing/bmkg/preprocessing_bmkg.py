from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import logging
import traceback
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BmkgPreprocessingError(Exception):
    """Custom exception for BMKG preprocessing errors"""
    pass

class BmkgDataValidator:
    """Validates BMKG data before preprocessing with various checks"""

    def validate_dataset(self, db, collection_name: str) -> Dict[str, Any]:
        """Enhanced validation with dataType, range, and temporal consistency checks"""
        try:
            # Check if collection exist
            if collection_name not in db.list_collection_names():
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} does not exist"]
                }
            
            # Get total records count
            total_records = db[collection_name].count_documents({})
            if total_records == 0:
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} is empty"]
                }
            
            logger.info(f"Validating {total_records} records in {collection_name}")

            # Sample validation - check 100 first records for performance
            sample_size = min(100, total_records)
            sample_docs = list(db[collection_name].find().limit(sample_size))

            # Define required fields with expected types and ranges
            field_requirements = {
                'Date': {'type': (str, pd.Timestamp, datetime), 'required': True, 'description': 'Date'},
                'TN': {'type': (int, float), 'range': (-10, 35), 'required': True, 'description': 'Minimum Temperature (°C)'},
                'TX': {'type': (int, float), 'range': (-5, 45), 'required': True, 'description': 'Maximum Temperature (°C)'},
                'TAVG': {'type': (int, float), 'range': (-5, 40), 'required': True, 'description': 'Average Temperature (°C)'},
                'RH_AVG': {'type': (int, float), 'range': (0, 100), 'required': True, 'description': 'Average Humidity (%)'},
                'RR': {'type': (int, float), 'range': (0, 500), 'required': True, 'description': 'Rainfall (mm)'},
                'SS': {'type': (int, float), 'range': (0, 14), 'required': True, 'description': 'Sunshine Duration (hours)'},
                'FF_X': {'type': (int, float), 'range': (0, 50), 'required': True, 'description': 'Maximum Wind Speed (m/s)'},
                'DDD_X': {'type': (int, float), 'range': (0, 360), 'required': True, 'description': 'Wind Direction (degrees)'},
                'FF_AVG': {'type': (int, float), 'range': (0, 30), 'required': True, 'description': 'Average Wind Speed (m/s)'},
                'DDD_CAR': {'type': str, 'required': True, 'description': 'Wind Direction (cardinal)'}
            }

            # Validation results
            validation_errors = []
            validation_warnings = []
            field_stats = {}

            # Check schema and data types
            first_doc = sample_docs[0]
            missing_fields = [field for field in field_requirements if field not in first_doc]

            if missing_fields:
                validation_errors.append(f"Missing required fields: {', '.join(missing_fields)}")

            # Validate each field in sample documents
            for field, requirements in field_requirements.items():
                if field not in first_doc:
                    continue
                
                field_values = []
                invalid_types = 0
                out_of_range = 0
                missing_values = 0

                for doc in sample_docs:
                    value = doc.get(field)

                    # Check for missing/null values
                    if value is None or value == '' or (isinstance(value, float) and np.isnan(value)):
                        missing_values += 1
                        continue

                    # Check for missing value codes (8888, 9999)
                    if isinstance(value, (int, float)) and value in [8888.0, 9999.0, 8888, 9999]:
                        missing_values += 1
                        continue

                    # Check data type
                    if not isinstance(value, requirements['type']):
                        invalid_types += 1
                        continue

                    # Check range for numeric fields
                    if 'range' in requirements and isinstance(value, (int, float)):
                        min_val, max_val = requirements['range']
                        if not (min_val <= value <= max_val):
                            out_of_range += 1

                    field_values.append(value)

                # Calculate statistics
                missing_pct = (missing_values / sample_size) * 100
                field_stats[field] = {
                    'missing_count': missing_values,
                    'missing_percentage': round(missing_pct, 2),
                    'invalid_types': invalid_types,
                    'out_of_range': out_of_range,
                    'valid_values': len(field_values),
                    'description': requirements.get('description', '')
                }

                # Generate warnings
                if missing_pct > 50:
                    validation_warnings.append(
                        f"Field '{field}' has {missing_pct:.1f}% missing values (high!)"
                    )
                elif missing_pct > 20:
                    validation_warnings.append(
                        f"Field '{field}' has {missing_pct:.1f}% missing values"
                    )

                if invalid_types > 0:
                    validation_errors.append(
                        f"Field '{field}' has {invalid_types} records with invalid data types"
                    )

                if out_of_range > sample_size * 0.1:
                    validation_warnings.append(
                        f"Field '{field}' has {out_of_range} outliers in sample"
                    )
            
            # Add physical relationship validation
            relationship_warnings = self._validate_physical_relationships(sample_docs)
            validation_warnings.extend(relationship_warnings)
                    
            # Validation date fields and temporal consistency
            date_validation = self._validate_temporal_continuity(db, collection_name, total_records)

            if not date_validation['valid']:
                validation_warnings.append(
                    f"Dataset has only {total_records} records. "
                    "ML models may not perform well with limited data."
                )
            
            # Determine if validation passed
            is_valid = len(validation_errors) == 0

            return {
                'valid': is_valid,
                'total_records': total_records,
                'sample_size': sample_size,
                'errors': validation_errors,
                'warnings': validation_warnings,
                'fields_statistics': field_stats,
                'temporal_info': date_validation.get('temporal_info', {}),
                'message': "Dataset validation completed" if is_valid else "Dataset validation failed"
            }
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }

    def _validate_temporal_continuity(self, db, collection_name: str, total_records: int) -> Dict[str, Any]:
        """Validate temporal continuity and coverage"""
        try:
            # Get first and latest dates
            first_record = db[collection_name].find_one(sort=[('Date', 1)])
            last_record = db[collection_name].find_one(sort=[('Date', -1)])
            
            if not first_record or not last_record:
                return {
                    'valid': False,
                    'errors': ["Cannot determine date range from dataset"]
                }
            
            # Extract dates
            first_date = first_record.get('Date')
            last_date = last_record.get('Date')
            
            # Convert to datetime if needed
            if isinstance(first_date, str):
                first_date = pd.to_datetime(first_date)
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
                
            # Calculate date range
            date_range = (last_date - first_date).days + 1
            
            warnings = []
            
            # Check if records count matches expected date range
            if total_records < date_range * 0.8:
                gap_pct = ((date_range - total_records) / date_range) * 100
                warnings.append(
                    f"Potential data gaps detected: Expected ~{date_range} records "
                    f"but found {total_records} records ({gap_pct:.1f}% gap)"
                )
            
            # Check for duplicates
            pipeline = [
                {'$group': {'_id': '$Date', 'count': {'$sum': 1}}},
                {'$match': {'count': {'$gt': 1}}}
            ]
            duplicates = list(db[collection_name].aggregate(pipeline))
            
            if duplicates:
                warnings.append(f"Found {len(duplicates)} dates with duplicate records")
            
            temporal_info = {
                'start_date': first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else str(first_date),
                'end_date': last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date),
                'date_range_days': date_range,
                'total_records': total_records,
                'coverage_percentage': round((total_records / date_range) * 100, 2),
                'duplicate_dates': len(duplicates)
            }
            
            return {
                'valid': True,
                'warnings': warnings,
                'temporal_info': temporal_info
            }
        except Exception as e:
            logger.error(f"Error validating temporal continuity: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Temporal validation error: {str(e)}"]
            }

    def _validate_physical_relationships(self, sample_docs: List[Dict]) -> List[str]:
        """Validate physical relationships in the data"""
        warnings = []
        
        temp_violations = 0
        wind_violations = 0
        
        for doc in sample_docs:
            # Check temperature relationships: TX >= TAVG >= TN
            tx = doc.get('TX')
            tn = doc.get('TN')
            tavg = doc.get('TAVG')
            
            if all(v is not None and not pd.isna(v) for v in [tx, tn, tavg]):
                if not (tn <= tavg <= tx):
                    temp_violations += 1
            
            # Check wind speed relationships: FF_X >= FF_AVG
            ff_x = doc.get('FF_X')
            ff_avg = doc.get('FF_AVG')
            
            if all(v is not None and not pd.isna(v) for v in [ff_x, ff_avg]):
                if ff_x < ff_avg:
                    wind_violations += 1
        
        if temp_violations > 0:
            warnings.append(f"Found {temp_violations} temperature relationship violations (TX >= TAVG >= TN)")
        
        if wind_violations > 0:
            warnings.append(f"Found {wind_violations} wind speed relationship violations (FF_X >= FF_AVG)")
        
        return warnings


class BmkgDataLoader:
    """Loads BMKG data from MongoDB into pandas DataFrame"""

    def load_data(self, db, collection_name: str) -> pd.DataFrame:
        """Load all data from MongoDB into pd.DataFrame"""
        try:
            # Load all records from MongoDB collection
            cursor = db[collection_name].find({})
            df = pd.DataFrame(list(cursor))

            if len(df) == 0:
                raise BmkgPreprocessingError(f"No data found in collection '{collection_name}'")

            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)

            # Ensure date columns is datetime type
            if 'Date' in df.columns and not pd.api.types.is_datetime64_dtype(df['Date']):
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.warning(f"Failed to convert 'Date' column to datetime: {str(e)}")

            # Sort by Date chronologically
            if 'Date' in df.columns:
                df = df.sort_values('Date')

            logger.info(f"Successfully loaded {len(df)} records from '{collection_name}'")
            return df
        except Exception as e:
            error_msg = f"Error loading data from collection '{collection_name}': {str(e)}"
            logger.error(error_msg)
            raise BmkgPreprocessingError(error_msg)


class BmkgDataSaver:
    """Saves preprocessed BMKG data back to new collection in MongoDB"""

    def save_preprocessed_data(
        self,
        db,
        preprocessed_data: pd.DataFrame,
        original_collection_name: str,
    ) -> Dict[str, Any]:
        """Save preprocessed data to a new collection, update dataset-meta"""
        try:
            # Generate cleaned collection name
            cleaned_collection_name = f"{original_collection_name}_cleaned"

            # Drop the existing cleaned collection if it exists
            if cleaned_collection_name in db.list_collection_names():
                logger.info(f"Dropping existing collection: {cleaned_collection_name}")
                db[cleaned_collection_name].drop()

            # Remove _id column - MongoDB will auto-generate new IDs
            if '_id' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('_id', axis=1)
            
            # Remove any existing __v column
            if '__v' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('__v', axis=1)
                logger.info("Dropped '__v' column from preprocessed data")
                
            # Drop temporary preprocessing columns
            temp_columns = ['Season', 'is_RR_missing', 'month']
            columns_to_drop = [col for col in temp_columns if col in preprocessed_data.columns]
            
            if columns_to_drop:
                preprocessed_data = preprocessed_data.drop(columns_to_drop, axis=1)
                logger.info(f"Dropped temporary columns: {columns_to_drop}")

            # Convert DataFrame records for MongoDB insertion
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
            raise BmkgPreprocessingError(error_msg)
        
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
            
            # Update metadata to point to cleaned collection
            update_fields = {
                "status": "preprocessed",
                "collectionName": cleaned_collection_name,
                "totalRecords": record_count,
                "columns": cleaned_columns,
                "lastUpdated": datetime.now(),
                "name": f"{original_meta.get('name', original_collection_name)} (Cleaned)"
            }
            
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


class BmkgPreprocessor:
    """Main class for preprocessing BMKG datasets with NASA-style coverage analysis"""

    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.validator = BmkgDataValidator()
        self.loader = BmkgDataLoader()
        self.saver = BmkgDataSaver()
        self.original_data = None
        
        # NASA-style preprocessing report structure
        self.preprocessing_report = {
            "missing_data": {},
            "outliers": {},
            "smoothing": {},
            "smoothing_validation": {},
            "gaps": {},
            "model_coverage": {},
            "quality_metrics": {},
            "warnings": []
        }

    def preprocess(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess BMKG dataset with NASA-style coverage analysis
        
        Args:
            options: Dictionary of preprocessing options
            
        Returns:
            Dictionary with preprocessing results and processed dataframe
        """
        try:
            start_time = datetime.now()
            
            # Default options (NASA-style)
            default_options = {
                "smoothing_method": "exponential",
                "window_size": 5,
                "exponential_alpha": 0.15,
                "adaptive_alpha_selection": True,
                "alpha_optimization_range": [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
                "drop_outliers": True,
                "outlier_methods": ["iqr", "zscore"],
                "iqr_multiplier": 2.0,
                "zscore_threshold": 3.5,
                "outlier_treatment": "interpolate",
                "fill_missing": True,
                "detect_gaps": True,
                "max_gap_interpolate": 30,  # 30 days for BMKG (vs 90 for NASA)
                "calculate_coverage": True,
                "columns_to_process": [
                    'TN', 'TX', 'TAVG', 'RH_AVG', 'RR',
                    'SS', 'FF_X', 'FF_AVG', 'DDD_X'
                ],
                "parameter_configs": {
                    "RR": {  
                    "smoothing_method": "exponential",
                    "exponential_alpha": 0.04, 
                    "apply_outlier_detection": True,
                    "outlier_treatment": "interpolate",
                    "iqr_multiplier": 2.5, 
                    "zscore_threshold": 3.5,
                    "preserve_zeros": False,
                    "validate_range": True,
                    "valid_min": 0.0,
                    "valid_max": 500.0,
                    "reason": "BMKG RR is noisy - α=0.04 smoothing + outlier detection balances denoising vs signal preservation"
                    },
                    "DDD_X": { 
                        "smoothing_method": None,
                        "apply_outlier_detection": True,
                        "reason": "Circular variable - exponential smoothing breaks on 0°-360° wraparound"
                    },
                    "SS": { 
                        "smoothing_method": "exponential",
                        "exponential_alpha": 0.12,
                        "reason": "Cloud patterns have inertia but vary daily"
                    },
                    "RH_AVG": {
                        "smoothing_method": "exponential",
                        "exponential_alpha": 0.12,  # Reduce from 0.15
                        "constrain_bounds": (0, 100),  # Apply bounds after smoothing
                        "reason": "Humidity is bounded variable"
                    },
                    "TAVG": {
                        "smoothing_method": "exponential",
                        "exponential_alpha": 0.12,
                        "reason": "Temperature is a critical variable"
                    }
                }
            }
            
            # Merge with provided options
            if options is None:
                options = {}
            self.options = {**default_options, **options}
            
            # Validate dataset
            logger.info("BMKG DATA PREPROCESSING PIPELINE")
            logger.info("\n[1/5] Validating dataset...")
            validation_result = self.validator.validate_dataset(self.db, self.collection_name)
            
            if not validation_result.get('valid', False):
                raise BmkgPreprocessingError(
                    f"Validation failed: {validation_result.get('errors', ['Unknown error'])}"
                )
            
            logger.info(f"✓ Validation passed - {validation_result['total_records']} records")
            
            # Load data
            logger.info("\n[2/5] Loading data from MongoDB...")
            df = self.loader.load_data(self.db, self.collection_name)
            original_record_count = len(df)
            logger.info(f"✓ Loaded {original_record_count} records")
            
            # Apply preprocessing
            logger.info("\n[3/5] Applying preprocessing...")
            processed_df = self._apply_preprocessing(df)
            logger.info("✓ Preprocessing completed")
            
            # Save processed data
            logger.info("\n[4/5] Saving preprocessed data...")
            save_result = self.saver.save_preprocessed_data(
                self.db,
                processed_df,
                self.collection_name
            )
            logger.info(f"✓ Saved to collection: {save_result['preprocessedCollections'][0]}")
            
            # Generate summary
            logger.info("\n[5/5] Generating summary...")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "status": "success",
                "message": "BMKG dataset preprocessed successfully",
                "collection": self.collection_name,
                "preprocessedData": processed_df.head(10).to_dict('records'),
                "recordCount": len(processed_df),
                "originalRecordCount": original_record_count,
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None,
                "processingTime": round(processing_time, 2),
                "preprocessing_report": self.preprocessing_report,
                "metadata": save_result.get("metadata", {})
            }
            
        except Exception as e:
            error_msg = f"Error preprocessing BMKG data: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise BmkgPreprocessingError(error_msg)

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NASA-style preprocessing pipeline to BMKG data"""
        logger.info("Starting BMKG data preprocessing with NASA-style pipeline...")
        
        processed_df = df.copy()
        total_steps = 10
        current_step = 0
        
        # Store original data BEFORE any modifications
        self.original_data = df.copy()
        
        def log_progress(stage, message):
            nonlocal current_step
            current_step += 1
            percentage = int((current_step / total_steps) * 100)
            logger.info(f"PROGRESS:{percentage}:{stage}:{message}")
        
        # Ensure Date is datetime and set as index
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'])
            processed_df = processed_df.set_index('Date').sort_index()
        
        # Add temporal features
        processed_df['month'] = processed_df.index.month
        processed_df['Season'] = processed_df.index.month.map(
            lambda m: 'Wet' if m in [9, 10, 11, 12, 1, 2, 3] else 'Dry'
        )
        
        # STEP 1: Replace BMKG missing codes
        log_progress("fill_values", "Replacing fill values with NaN...")
        processed_df = self._replace_fill_values(processed_df)
        
        # STEP 2: Detect and report gaps
        log_progress("gap_detection", "Detecting gaps in time series...")
        if self.options.get("detect_gaps", True):
            self._detect_gaps(processed_df)
        
        # STEP 3: Handle missing values (BMKG-specific imputation)
        log_progress("imputation", "Imputing missing values...")
        if self.options.get("fill_missing", True):
            processed_df = self._impute_missing_values(processed_df)
        
        # STEP 4: Detect and handle outliers
        log_progress("outliers", "Detecting and handling outliers...")
        if self.options.get("drop_outliers", True):
            processed_df = self._handle_outliers(processed_df)
        
        # STEP 5: Apply physical constraints
        log_progress("physical_constraints", "Applying physical constraints...")
        processed_df = self._apply_physical_constraints(processed_df)
        
        # STEP 6: Optimize alpha (if enabled)
        if self.options.get("adaptive_alpha_selection", False):
            log_progress("alpha_optimization", "Optimizing smoothing parameters...")
            self._optimize_parameter_alphas(processed_df)
        
        # STEP 7: Apply smoothing
        log_progress("smoothing", "Applying smoothing methods...")
        processed_df = self._apply_smoothing(processed_df)
        
        # STEP 8: Validate smoothing quality
        log_progress("smoothing_validation", "Validating smoothing quality (GCV + Trend)...")
        self._validate_smoothing_method(processed_df)
        
        # STEP 9: Calculate model coverage
        log_progress("model_coverage", "Calculating model coverage analysis...")
        if self.options.get("calculate_coverage", True):
            self._calculate_model_coverage(processed_df)
        
        # STEP 10: Generate quality metrics
        log_progress("quality_metrics", "Generating quality metrics...")
        self._generate_quality_metrics(df, processed_df)
        
        # Reset index to have Date as column
        processed_df = processed_df.reset_index()
        
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df

    def _replace_fill_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace BMKG fill values (8888, 9999) with NaN"""
        logger.info("Replacing fill values with NaN...")
        
        fill_values = [8888, 9999, 8888.0, 9999.0]
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

    def _detect_gaps(self, df: pd.DataFrame) -> None:
        """Detect and classify gaps in the BMKG time series"""
        logger.info("Detecting gaps in time series data...")
        
        dates = df.index
        
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
        
        # Classify gaps (BMKG thresholds)
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
            "gap_details": gap_details[:10]  # First 10
        }
        logger.info(f"Detected {len(gaps)} gaps: {small_gaps} small, {medium_gaps} medium, {large_gaps} large")
   
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using BMKG-specific methods"""
        logger.info("Imputing missing values...")
        
        # Order matters! Temperature and humidity first (needed for rainfall model)
        df = self._impute_temperature(df)
        df = self._impute_humidity(df)
        df = self._impute_rainfall(df)
        df = self._impute_wind_speed_max(df)
        df = self._impute_wind_speed_avg(df)
        df = self._impute_wind_direction_degrees(df)
        df = self._impute_sunshine_duration(df)
        df = self._impute_wind_direction_cardinal(df)
        
        return df
    
    def _impute_rainfall(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Impute missing rainfall (RR) values using context-aware methods
        BMKG RR special handling:
        - Preserves zero values (dry days are important signal, not missing data)
        - Uses surrounding context for interpolation
        - Avoids creating unrealistic rain patterns
        """
        
        if 'RR' not in df.columns:
            return df
        
        missing_mask = df['RR'].isna()
        
        if not missing_mask.any():
            logger.info("  ✓ RR: No missing values")
            return df
        
        missing_count = missing_mask.sum()
        logger.info(f"  • RR: Imputing {missing_count} missing values...")
        
        # Get indices of missing values
        missing_indices = np.where(missing_mask)[0]
        
        imputed_count = 0
        
        for idx in missing_indices:
            # Get surrounding context (before and after)
            before_idx = max(0, idx - 1)
            after_idx = min(len(df) - 1, idx + 1)
            
            before_val = df.iloc[before_idx]['RR'] if before_idx < idx else np.nan
            after_val = df.iloc[after_idx]['RR'] if after_idx > idx else np.nan
            
            if pd.notna(before_val) and pd.notna(after_val):
                # Both sides have data
                if before_val == 0 and after_val == 0:
                    # No rain before and after → assume no rain (dry period)
                    df.loc[df.index[idx], 'RR'] = 0.0
                else:
                    # Interpolate between rainy periods
                    df.loc[df.index[idx], 'RR'] = (before_val + after_val) / 2
                imputed_count += 1
                
            elif pd.notna(before_val):
                # Only before has data → forward fill
                df.loc[df.index[idx], 'RR'] = before_val
                imputed_count += 1
                
            elif pd.notna(after_val):
                # Only after has data → backward fill
                df.loc[df.index[idx], 'RR'] = after_val
                imputed_count += 1
        
        # Log imputation results
        if imputed_count > 0:
            logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} RR values")
        
        # Store imputation stats
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["RR"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "context_aware"
        }
        
        return df
    
    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing temperature values (TN, TX, TAVG)
        Uses standard linear + spline interpolation
        """
        temp_params = ['TN', 'TX', 'TAVG']
        
        for param in temp_params:
            if param not in df.columns:
                continue
            
            missing_count = df[param].isna().sum()
            
            if missing_count == 0:
                logger.info(f"  ✓ {param}: No missing values")
                continue
            
            logger.info(f"  • {param}: Imputing {missing_count} missing values...")
            
            # Step 1: Linear interpolation (7 day limit)
            df[param] = df[param].interpolate(method='linear', limit=7, limit_direction='both')
            
            # Step 2: Spline for remaining gaps (30 day limit)
            remaining_missing = df[param].isna().sum()
            if remaining_missing > 0 and remaining_missing < missing_count:
                df[param] = df[param].interpolate(method='spline', order=3, limit=30, limit_direction='both')
            
            # Step 3: Forward/backward fill for edge cases (3 day limit)
            df[param] = df[param].fillna(method='ffill', limit=3)
            df[param] = df[param].fillna(method='bfill', limit=3)
            
            final_missing = df[param].isna().sum()
            imputed_count = missing_count - final_missing
            
            logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} {param} values")
            
            if "imputation" not in self.preprocessing_report:
                self.preprocessing_report["imputation"] = {}
            self.preprocessing_report["imputation"][param] = {
                "missing_count": int(missing_count),
                "imputed_count": int(imputed_count),
                "method": "linear_spline_ffill"
            }
        
        return df

    def _impute_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing humidity values (RH_AVG)
        Uses standard interpolation with bounds enforcement
        """
        if 'RH_AVG' not in df.columns:
            return df
        
        missing_count = df['RH_AVG'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ RH_AVG: No missing values")
            return df
        
        logger.info(f"  • RH_AVG: Imputing {missing_count} missing values...")
        
        # Linear interpolation
        df['RH_AVG'] = df['RH_AVG'].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline for remaining gaps
        remaining_missing = df['RH_AVG'].isna().sum()
        if remaining_missing > 0:
            df['RH_AVG'] = df['RH_AVG'].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Enforce valid range (0-100%)
        df['RH_AVG'] = df['RH_AVG'].clip(0, 100)
        
        final_missing = df['RH_AVG'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} RH_AVG values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["RH_AVG"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "linear_spline_clip"
        }
        
        return df
    
    def _impute_wind_speed_max(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing max wind speed values (FF_X)
        """
        if 'FF_X' not in df.columns:
            return df
        
        missing_count = df['FF_X'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ FF_X: No missing values")
            return df
        
        logger.info(f"  • FF_X: Imputing {missing_count} missing values...")
        
        # Linear interpolation
        df['FF_X'] = df['FF_X'].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline for remaining gaps
        remaining_missing = df['FF_X'].isna().sum()
        if remaining_missing > 0:
            df['FF_X'] = df['FF_X'].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Enforce non-negative
        df['FF_X'] = df['FF_X'].clip(lower=0)
        
        final_missing = df['FF_X'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} FF_X values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["FF_X"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "linear_spline_clip"
        }
        
        return df
    
    def _impute_wind_speed_avg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing average wind speed values (FF_AVG)
        """
        if 'FF_AVG' not in df.columns:
            return df
        
        missing_count = df['FF_AVG'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ FF_AVG: No missing values")
            return df
        
        logger.info(f"  • FF_AVG: Imputing {missing_count} missing values...")
        
        # Linear interpolation
        df['FF_AVG'] = df['FF_AVG'].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline for remaining gaps
        remaining_missing = df['FF_AVG'].isna().sum()
        if remaining_missing > 0:
            df['FF_AVG'] = df['FF_AVG'].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Enforce non-negative
        df['FF_AVG'] = df['FF_AVG'].clip(lower=0)
        
        final_missing = df['FF_AVG'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} FF_AVG values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["FF_AVG"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "linear_spline_clip"
        }
        
        return df
    
    def _impute_wind_direction_degrees(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing wind direction (degrees) values (DDD_X)
        Handles circular nature of 0°-360° data
        """
        if 'DDD_X' not in df.columns:
            return df
        
        missing_count = df['DDD_X'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ DDD_X: No missing values")
            return df
        
        logger.info(f"  • DDD_X: Imputing {missing_count} missing values...")
        
        # For small gaps, use linear interpolation (won't preserve circularity but simple)
        df['DDD_X'] = df['DDD_X'].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline for remaining gaps
        remaining_missing = df['DDD_X'].isna().sum()
        if remaining_missing > 0:
            df['DDD_X'] = df['DDD_X'].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Forward/backward fill for edge cases
        df['DDD_X'] = df['DDD_X'].fillna(method='ffill', limit=3)
        df['DDD_X'] = df['DDD_X'].fillna(method='bfill', limit=3)
        
        # Ensure within valid range (0-360 degrees)
        df['DDD_X'] = df['DDD_X'].clip(0, 360)
        # Handle wraparound (values > 360 should wrap, values < 0 should wrap)
        df['DDD_X'] = df['DDD_X'] % 360
        
        final_missing = df['DDD_X'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} DDD_X values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["DDD_X"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "linear_spline_circular"
        }
        
        return df
    
    def _impute_sunshine_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing sunshine duration values (SS)
        SS ranges from 0-14 hours
        """
        if 'SS' not in df.columns:
            return df
        
        missing_count = df['SS'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ SS: No missing values")
            return df
        
        logger.info(f"  • SS: Imputing {missing_count} missing values...")
        
        # Linear interpolation
        df['SS'] = df['SS'].interpolate(method='linear', limit=7, limit_direction='both')
        
        # Spline for remaining gaps
        remaining_missing = df['SS'].isna().sum()
        if remaining_missing > 0:
            df['SS'] = df['SS'].interpolate(method='spline', order=3, limit=30, limit_direction='both')
        
        # Enforce valid range (0-14 hours)
        df['SS'] = df['SS'].clip(0, 14)
        
        final_missing = df['SS'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} SS values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["SS"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "linear_spline_clip"
        }
        
        return df
    
    def _impute_wind_direction_cardinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing cardinal wind direction values (DDD_CAR)
        Categorical data: N, NE, E, SE, S, SW, W, NW, C
        """
        if 'DDD_CAR' not in df.columns:
            return df
        
        missing_count = df['DDD_CAR'].isna().sum()
        
        if missing_count == 0:
            logger.info(f"  ✓ DDD_CAR: No missing values")
            return df
        
        logger.info(f"  • DDD_CAR: Imputing {missing_count} missing values...")
        
        # For categorical data, use forward/backward fill
        df['DDD_CAR'] = df['DDD_CAR'].fillna(method='ffill', limit=3)
        df['DDD_CAR'] = df['DDD_CAR'].fillna(method='bfill', limit=3)
        
        # If still missing, use mode (most common direction)
        remaining_missing = df['DDD_CAR'].isna().sum()
        if remaining_missing > 0:
            mode_direction = df['DDD_CAR'].mode()[0] if len(df['DDD_CAR'].mode()) > 0 else 'C'  # 'C' = Calm
            df['DDD_CAR'].fillna(mode_direction, inplace=True)
        
        final_missing = df['DDD_CAR'].isna().sum()
        imputed_count = missing_count - final_missing
        
        logger.info(f"    ✓ Imputed {imputed_count}/{missing_count} DDD_CAR values")
        
        if "imputation" not in self.preprocessing_report:
            self.preprocessing_report["imputation"] = {}
        self.preprocessing_report["imputation"]["DDD_CAR"] = {
            "missing_count": int(missing_count),
            "imputed_count": int(imputed_count),
            "method": "ffill_bfill_mode"
        }
        
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using IQR and Z-score methods"""
        logger.info("Detecting and handling outliers...")
        
        params = self.options.get("columns_to_process", [])
        outlier_methods = self.options.get("outlier_methods", ["iqr", "zscore"])
        outlier_summary = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Skip outlier detection for specific parameters
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
        
        multiplier = self.options.get("iqr_multiplier", 2.0)
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
        threshold = self.options.get("zscore_threshold", 3.5)
        
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
            multiplier = self.options.get("iqr_multiplier", 2.0)
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df.loc[df[param] < lower_bound, param] = lower_bound
            df.loc[df[param] > upper_bound, param] = upper_bound
        
        elif treatment == "remove":
            # Remove rows with outliers (not recommended)
            df = df[~outliers_mask]
        
        return df

    def _apply_physical_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physical constraints to ensure data consistency"""
        logger.info("Applying physical constraints...")
        
        violations_fixed = {
            "temp_tavg_gt_tx": 0,
            "temp_tavg_lt_tn": 0,
            "wind_ffx_lt_ffavg": 0,
            "humidity_out_of_bounds": 0,
            "negative_rainfall": 0,
            "negative_sunshine": 0
        }
        
        # Temperature constraints: TX >= TAVG >= TN
        if all(col in df.columns for col in ['TX', 'TAVG', 'TN']):
            # Fix TAVG > TX
            violation_mask = df['TAVG'] > df['TX']
            if violation_mask.sum() > 0:
                violations_fixed["temp_tavg_gt_tx"] = int(violation_mask.sum())
                df.loc[violation_mask, 'TAVG'] = df.loc[violation_mask, 'TX']

            # Fix TAVG < TN
            violation_mask = df['TAVG'] < df['TN']
            if violation_mask.sum() > 0:
                violations_fixed["temp_tavg_lt_tn"] = int(violation_mask.sum())
                df.loc[violation_mask, 'TAVG'] = df.loc[violation_mask, 'TN']

        # Wind speed constraints: FF_X >= FF_AVG
        if 'FF_X' in df.columns and 'FF_AVG' in df.columns:
            violation_mask = df['FF_X'] < df['FF_AVG']
            if violation_mask.sum() > 0:
                violations_fixed["wind_ffx_lt_ffavg"] = int(violation_mask.sum())
                df.loc[violation_mask, 'FF_X'] = df.loc[violation_mask, 'FF_AVG'] * 1.2

        # Humidity bounds
        if 'RH_AVG' in df.columns:
            before_clip = ((df['RH_AVG'] < 0) | (df['RH_AVG'] > 100)).sum()
            df['RH_AVG'] = df['RH_AVG'].clip(0, 100)
            violations_fixed["humidity_out_of_bounds"] = int(before_clip)

        # Non-negative rainfall and sunshine
        if 'RR' in df.columns:
            negative_rr = (df['RR'] < 0).sum()
            df['RR'] = df['RR'].clip(lower=0)
            violations_fixed["negative_rainfall"] = int(negative_rr)

        if 'SS' in df.columns:
            negative_ss = (df['SS'] < 0).sum()
            df['SS'] = df['SS'].clip(lower=0)
            violations_fixed["negative_sunshine"] = int(negative_ss)
        
        self.preprocessing_report["physical_constraints"] = violations_fixed
        logger.info(f"Physical constraints applied: {sum(violations_fixed.values())} violations fixed")
        return df

    def _optimize_parameter_alphas(self, df: pd.DataFrame) -> None:
        """Optimize alpha per parameter using GCV + trend preservation"""
        logger.info("🔍 Starting adaptive alpha optimization...")
        
        params = self.options.get("columns_to_process", [])
        alpha_range = self.options.get("alpha_optimization_range", [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20])
        
        optimization_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            smoothing_method = param_config.get(
                "smoothing_method",
                self.options.get("smoothing_method", "exponential")
            )
            
            # Skip if no smoothing or non-exponential
            if smoothing_method is None:
                logger.info(f"  ⏭️  {param}: Skipping (no smoothing configured)")
                continue
            
            if smoothing_method != "exponential":
                logger.info(f"  ⏭️  {param}: Skipping (using {smoothing_method} method)")
                continue
            
            optimal_alpha = self._find_optimal_alpha_for_parameter(df, param, alpha_range)
            
            # Store in config
            if "parameter_configs" not in self.options:
                self.options["parameter_configs"] = {}
            if param not in self.options["parameter_configs"]:
                self.options["parameter_configs"][param] = {}
            
            self.options["parameter_configs"][param]["exponential_alpha"] = optimal_alpha
            optimization_results[param] = optimal_alpha
            
            logger.info(f"  ✓ {param}: Optimal α = {optimal_alpha:.3f}")
        
        self.preprocessing_report["alpha_optimization"] = optimization_results
        logger.info(f"Alpha optimization complete: {len(optimization_results)} parameters optimized")

    def _find_optimal_alpha_for_parameter(
        self,
        df: pd.DataFrame,
        param: str,
        alpha_range: List[float]
    ) -> float:
        """Test multiple alpha values and return the optimal one"""
        
        if param not in df.columns:
            return self.options.get("exponential_alpha", 0.15)
        
        series = df[param].dropna()
        
        if len(series) < 100:
            logger.warning(f"  {param}: Insufficient data ({len(series)} points) - using default alpha")
            return self.options.get("exponential_alpha", 0.15)
        
        best_alpha = self.options.get("exponential_alpha", 0.15)
        best_score = -np.inf
        test_results = []
        
        logger.info(f"     Testing {len(alpha_range)} alpha values...")
        
        for alpha in alpha_range:
            performance = self._test_alpha_performance(series.values, alpha)
            
            if performance is None:
                continue
            
            gcv_score = performance.get("gcv_score", 10.0)
            trend_pct = performance.get("trend_preservation_pct", 0.0)
            
            # Scoring: prioritize trend preservation
            combined_score = (trend_pct * 1.0) - (gcv_score * 0.5)
            
            test_results.append({
                "alpha": alpha,
                "gcv": gcv_score,
                "trend": trend_pct,
                "score": combined_score
            })
            
            logger.info(f"     α={alpha}: GCV={gcv_score:.4f}, Trend={trend_pct:.1f}%, Score={combined_score:.2f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_alpha = alpha
        
        if test_results:
            best_result = max(test_results, key=lambda x: x["score"])
            logger.info(f"     🏆 BEST: α={best_result['alpha']}, Trend={best_result['trend']:.1f}%")
        
        return best_alpha

    def _test_alpha_performance(
        self,
        original_series: np.ndarray,
        alpha: float
    ) -> Optional[Dict[str, Any]]:
        """Test a single alpha value and return GCV + trend preservation metrics"""
        
        try:
            if len(original_series) < 100:
                return None
            
            # Check NaN issues
            nan_count = np.isnan(original_series).sum()
            if nan_count == len(original_series) or nan_count > len(original_series) * 0.5:
                return None
            
            if alpha <= 0 or alpha >= 1:
                return None
            
            # Apply exponential smoothing
            series_pd = pd.Series(original_series).dropna()
            
            if len(series_pd) < 100:
                return None
            
            smoothed_series = series_pd.ewm(alpha=alpha, adjust=False).mean().values
            original_clean = original_series[~np.isnan(original_series)]
            
            # Ensure same length
            min_len = min(len(original_clean), len(smoothed_series))
            original_clean = original_clean[:min_len]
            smoothed_series = smoothed_series[:min_len]
            
            gcv_score = self._calculate_gcv(original_clean, smoothed_series, param=None)
            trend_preservation = self._calculate_trend_preservation(original_clean, smoothed_series)
            
            return {
                "alpha": alpha,
                "gcv_score": gcv_score,
                "trend_preservation_pct": trend_preservation * 100,
                "combined_score": (trend_preservation * 100) - (gcv_score * 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Error testing α={alpha}: {str(e)}")
            return None

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
                # Skip smoothing for this parameter
                smoothing_summary[param] = "none"
                continue
            
            # Apply smoothing
            if param_smoothing == "moving_average":
                df[param] = self._smooth_moving_average(df[param])
                smoothing_summary[param] = "moving_average"
            elif param_smoothing == "exponential":
                df[param] = self._smooth_exponential(df[param], param)
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

    def _smooth_exponential(
        self,
        series: pd.Series,
        param: str = None
    ) -> pd.Series:
        """Apply exponential smoothing with parameter-specific alpha"""
        
        if param:
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
        else:
            alpha = self.options.get("exponential_alpha", 0.15)
        
        return series.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_gcv(self, original: np.ndarray, smoothed: np.ndarray, param: str) -> float:
        """Calculate Generalized Cross-Validation score"""
        n = len(original)
        mse = np.mean((original - smoothed) ** 2)
        
        # Get parameter-specific smoothing method
        if param:
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            smoothing_method = param_config.get(
                "smoothing_method",
                self.options.get("smoothing_method", "exponential")
            )
        else:
            smoothing_method = self.options.get("smoothing_method", "exponential")
        
        if smoothing_method is None:
            return 0.0
        
        # Estimate effective degrees of freedom
        if smoothing_method == "moving_average":
            window_size = self.options.get("window_size", 5)
            edf = window_size
        elif smoothing_method == "exponential":
            if param:
                param_config = self.options.get("parameter_configs", {}).get(param, {})
                alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
            else:
                alpha = self.options.get("exponential_alpha", 0.15)
            
            alpha = max(0.01, min(alpha, 0.99))
            edf = 1.0 / (2.0 - alpha)
            edf = max(1.0, min(edf, n / 3.0))
        else:
            edf = 5
        
        # GCV formula
        denominator = (1.0 - edf / n) ** 2
        denominator = max(denominator, 0.01)
        
        gcv = mse / denominator
        return gcv

    def _calculate_trend_preservation(self, original: np.ndarray, smoothed: np.ndarray) -> float:
        """Calculate trend direction agreement between original and smoothed data"""
        
        if len(original) != len(smoothed):
            logger.warning(f"Trend preservation: length mismatch")
            return np.nan
        
        if len(original) < 2:
            return np.nan
        
        # Calculate first differences
        original_diff = np.diff(original)
        smoothed_diff = np.diff(smoothed)
        
        # Get direction signs
        original_direction = np.sign(original_diff)
        smoothed_direction = np.sign(smoothed_diff)
        
        # Exclude flat regions
        non_zero_mask = (original_direction != 0) & (smoothed_direction != 0)
        
        if non_zero_mask.sum() == 0:
            # All flat = perfect preservation
            return 1.0
        
        if non_zero_mask.sum() < 10:
            logger.debug(f"Very few trend changes ({non_zero_mask.sum()}) - may be unreliable")
        
        # Calculate agreement
        agreement = (original_direction[non_zero_mask] == smoothed_direction[non_zero_mask])
        trend_preservation = agreement.mean()
        
        return trend_preservation

    def _validate_smoothing_method(self, df: pd.DataFrame) -> None:
        """
        Validate smoothing quality using GCV + Trend Preservation
        """
        logger.info("Validating smoothing quality...")
        
        params = self.options.get("columns_to_process", [])
        validation_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Check if smoothing was applied
            smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})
            param_smoothing_applied = smoothing_summary.get(param, "none")
            
            if param_smoothing_applied == "none":
                validation_results[param] = {
                    "status": "no_smoothing",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            
            if 'Date' not in self.original_data.columns:
                logger.warning(f"Original data missing Date column for {param}")
                validation_results[param] = {
                    "status": "missing_date_column_original",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            
            if 'Date' not in df.columns:
                logger.warning(f"Processed data missing Date column for {param}")
                validation_results[param] = {
                    "status": "missing_date_column_processed",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            
            try:
                original = self.original_data.set_index('Date')[param].dropna()    
                smoothed = df.set_index('Date')[param].dropna()    
                
                # Align on Date index
                common_idx = original.index.intersection(smoothed.index)
                
                if len(common_idx) == 0:
                    logger.warning(f"No date overlap found for {param}")
                    validation_results[param] = {
                        "status": "no_date_overlap",
                        "gcv": None,
                        "trend_preservation": None
                    }
                    continue
                
                # Extract aligned values
                original_aligned = original.loc[common_idx].values
                smoothed_aligned = smoothed.loc[common_idx].values
                
                # Check minimum data requirement
                if len(common_idx) < 30:
                    validation_results[param] = {
                        "status": "insufficient_data",
                        "data_points": len(common_idx),
                        "gcv": None,
                        "trend_preservation": None
                    }
                    continue
                
                # Calculate metrics
                gcv_score = self._calculate_gcv(original_aligned, smoothed_aligned, param)
                trend_agreement = self._calculate_trend_preservation(original_aligned, smoothed_aligned)
                
                # Determine quality
                quality_status = self._determine_smoothing_quality(gcv_score, trend_agreement)
                
                validation_results[param] = {
                    "gcv_score": round(float(gcv_score), 4),
                    "trend_preservation_pct": round(float(trend_agreement * 100), 2),
                    "quality_status": quality_status,
                    "smoothing_method": param_smoothing_applied,
                    "data_points": len(common_idx)
                }
                
                logger.info(
                    f"  {param}: GCV={gcv_score:.4f}, Trend={trend_agreement*100:.2f}%, Quality={quality_status.upper()}"
                )
                
                # Add warnings for poor quality
                if quality_status == "poor":
                    self.preprocessing_report["warnings"].append(
                        f"Parameter {param}: smoothing quality is poor "
                        f"(GCV={gcv_score:.3f}, Trend={trend_agreement*100:.1f}%)"
                    )
            
            except Exception as e:
                logger.error(f"Error validating smoothing for {param}: {str(e)}")
                validation_results[param] = {
                    "status": "error",
                    "error": str(e),
                    "gcv": None,
                    "trend_preservation": None
                }
        
        self.preprocessing_report["smoothing_validation"] = validation_results
        logger.info(f"Smoothing validation completed for {len(validation_results)} parameters")

    def _determine_smoothing_quality(self, gcv: float, trend_preservation: float) -> str:
        """Determine overall smoothing quality"""
        if gcv < 2.0 and trend_preservation > 0.80:
            return "excellent"
        elif gcv < 4.0 and trend_preservation > 0.75:
            return "good"
        elif gcv < 10.0 and trend_preservation > 0.70:
            return "fair"
        else:
            return "poor"

    def _calculate_model_coverage(self, df: pd.DataFrame) -> None:
        """Calculate model applicability coverage for Holt-Winters and LSTM models"""
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
        
        self.preprocessing_report["model_coverage"] = coverage_results
        logger.info(f"Model coverage - Holt Winters: {coverage_results['holt_winters']['coverage_percentage']:.1f}%")
        logger.info(f"Model coverage - LSTM: {coverage_results['lstm']['coverage_percentage']:.1f}%")

    def _analyze_parameter_coverage(
        self,
        df: pd.DataFrame,
        param: str,
        total_points: int
    ) -> Dict[str, Any]:
        """Analyze coverage for a specific parameter for both model types"""
        
        series = df[param].dropna()
        if len(series) < 30:
            return {
                "holt_winters_coverage": 0,
                "lstm_coverage": 0,
                "insufficient_data": True
            }
        
        coverage_analysis = {
            "data_points": len(series),
            "missing_ratio": (total_points - len(series)) / total_points
        }
        
        # Run all analyses
        large_gaps_impact = self._analyze_large_gaps(df, param)
        extreme_outliers_impact = self._analyze_extreme_outliers(series)
        seasonality_analysis = self._analyze_seasonality(series)
        stationarity_analysis = self._analyze_stationarity(series)
        rainfall_analysis = self._analyze_rainfall_extremes(series, param) if param == "RR" else {}
        
        # Check if parameter was smoothed
        smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})
        was_smoothed = smoothing_summary.get(param, "none") != "none"
        
        # Calculate coverage
        if was_smoothed:
            hw_coverage = self._calculate_holt_winters_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                seasonality_analysis,
                stationarity_analysis,
                coverage_analysis,
                param
            )
            
            lstm_coverage = self._calculate_lstm_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                rainfall_analysis,
                stationarity_analysis,
                coverage_analysis,
                param
            )
        else:
            # Non-smoothed parameters
            non_smoothed_coverage = self._calculate_non_smoothed_coverage(
                large_gaps_impact,
                extreme_outliers_impact,
                coverage_analysis,
                param
            )
            hw_coverage = lstm_coverage = non_smoothed_coverage
        
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
                "rainfall": rainfall_analysis
            }
        }

    def _analyze_large_gaps(self, df: pd.DataFrame, param: str) -> Dict[str, Any]:
        """Analyze impact of large gaps (>30 days for BMKG)"""
        
        gaps_info = self.preprocessing_report.get("gaps", {})
        large_gaps = [gap for gap in gaps_info.get("gap_details", []) if gap.get("duration_days", 0) > 30]
        
        if not large_gaps:
            return {"impact_percentage": 0, "large_gaps_count": 0}
        
        total_gap_days = sum(gap["duration_days"] for gap in large_gaps)
        total_days = (df.index.max() - df.index.min()).days + 1
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
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return {"impact_percentage": 0}
        
        z_scores = np.abs((series - mean) / std)
        extreme_outliers = z_scores > 3.5
        
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
            
    def _analyze_rainfall_extremes(self, series: pd.Series, param: str) -> Dict[str, Any]:
            """Analyze extreme precipitation events (0 vs 500mm range)"""
            if param != "RR":
                return {}
            
            zero_ratio = (series == 0).sum() / len(series)
            extreme_high = (series > 200).sum() / len(series) 
            # Calculate range ratio (wide range affects LSTM)
            data_range = series.max() - series.min()
            range_impact = min(data_range / 300, 1.0) 
            
            return {
                "zero_precipitation_ratio": round(float(zero_ratio), 3),
                "extreme_precipitation_ratio": round(float(extreme_high), 3),
                "range_impact": round(float(range_impact), 3),
                "max_precipitation": round(float(series.max()), 2)
            }
    def log_detailed_coverage_analysis(self, processed_df: pd.DataFrame) -> None:
        """
        Log detailed coverage analysis for debugging and verification.
        Outputs comprehensive breakdown of penalties, thresholds, and quality metrics.
        """
        logger.info("="*80)
        logger.info("DETAILED COVERAGE ANALYSIS REPORT")
        logger.info("="*80)
        
        # Get coverage data
        coverage_data = self.preprocessing_report.get("model_coverage", {})
        per_param = coverage_data.get("per_parameter", {})
        
        if not per_param:
            logger.warning("No per-parameter coverage data available")
            return
        
        # Overall Summary
        logger.info("\n📊 OVERALL COVERAGE SUMMARY:")
        logger.info(f"  Holt-Winters: {coverage_data.get('holt_winters', {}).get('coverage_percentage', 0):.2f}%")
        logger.info(f"  LSTM: {coverage_data.get('lstm', {}).get('coverage_percentage', 0):.2f}%")
        logger.info(f"  HW Suitability: {coverage_data.get('holt_winters', {}).get('model_suitability', 'unknown').upper()}")
        logger.info(f"  LSTM Suitability: {coverage_data.get('lstm', {}).get('model_suitability', 'unknown').upper()}")
        
        # Per-Parameter Detailed Breakdown
        logger.info("\n" + "="*80)
        logger.info("PER-PARAMETER DETAILED ANALYSIS:")
        logger.info("="*80)
        
        params = self.options.get("columns_to_process", [])
        
        for param in params:
            if param not in per_param:
                logger.info(f"\n❌ {param}: No coverage data available")
                continue
            
            param_data = per_param[param]
            
            # Header
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 PARAMETER: {param}")
            logger.info(f"{'='*80}")
            
            # Basic Coverage
            hw_coverage = param_data.get("holt_winters_coverage", 0)
            lstm_coverage = param_data.get("lstm_coverage", 0)
            logger.info(f"  📈 Coverage:")
            logger.info(f"     • Holt-Winters: {hw_coverage:.2f}%")
            logger.info(f"     • LSTM: {lstm_coverage:.2f}%")
            
            # Analysis Details
            analysis = param_data.get("analysis_details", {})
            
            # 1. Smoothing Quality (if available)
            smoothing_validation = self.preprocessing_report.get("smoothing_validation", {})
            if param in smoothing_validation:
                smooth_data = smoothing_validation[param]
                logger.info(f"\n  🎯 SMOOTHING QUALITY:")
                logger.info(f"     • Status: {smooth_data.get('quality_status', 'unknown').upper()}")
                logger.info(f"     • GCV Score: {smooth_data.get('gcv_score', 'N/A')}")
                logger.info(f"     • Trend Preservation: {smooth_data.get('trend_preservation_pct', 'N/A')}%")
                logger.info(f"     • Method: {smooth_data.get('smoothing_method', 'none')}")
                logger.info(f"     • Data Points: {smooth_data.get('data_points', 0)}")
            
            # 2. Large Gaps Analysis
            large_gaps = analysis.get("large_gaps", {})
            if large_gaps:
                logger.info(f"\n  📏 LARGE GAPS ANALYSIS:")
                logger.info(f"     • Impact: {large_gaps.get('impact_percentage', 0):.2f}%")
                logger.info(f"     • Count: {large_gaps.get('large_gaps_count', 0)}")
                logger.info(f"     • Total Gap Days: {large_gaps.get('total_gap_days', 0)}")
            else:
                logger.info(f"\n  ✅ LARGE GAPS: None detected")
            
            # 3. Extreme Outliers Analysis
            outliers = analysis.get("extreme_outliers", {})
            if outliers:
                logger.info(f"\n  ⚠️  EXTREME OUTLIERS:")
                logger.info(f"     • Impact: {outliers.get('impact_percentage', 0):.2f}%")
                logger.info(f"     • Count: {outliers.get('extreme_outliers_count', 0)}")
                logger.info(f"     • Max Z-Score: {outliers.get('max_z_score', 'N/A')}")
            else:
                logger.info(f"\n  ✅ EXTREME OUTLIERS: None detected")
            
            # 4. Seasonality Analysis
            seasonality = analysis.get("seasonality", {})
            if seasonality and not seasonality.get("insufficient_data", False):
                logger.info(f"\n  🌊 SEASONALITY ANALYSIS:")
                logger.info(f"     • Seasonal Strength: {seasonality.get('seasonal_strength', 0):.3f}")
                logger.info(f"     • Has Clear Seasonality: {'YES' if seasonality.get('has_clear_seasonality', False) else 'NO'}")
                logger.info(f"     • Seasonal Variance: {seasonality.get('seasonal_variance', 'N/A')}")
                logger.info(f"     • Residual Variance: {seasonality.get('residual_variance', 'N/A')}")
                if seasonality.get("error"):
                    logger.info(f"     • Error: {seasonality.get('error')}")
            else:
                logger.info(f"\n  ⚠️  SEASONALITY: Insufficient data or not analyzed")
            
            # 5. Stationarity Analysis
            stationarity = analysis.get("stationarity", {})
            if stationarity and not stationarity.get("insufficient_data", False):
                logger.info(f"\n  📊 STATIONARITY ANALYSIS:")
                logger.info(f"     • Is Stationary: {'YES' if stationarity.get('is_stationary', False) else 'NO'}")
                logger.info(f"     • ADF Statistic: {stationarity.get('adf_statistic', 'N/A')}")
                logger.info(f"     • P-Value: {stationarity.get('p_value', 'N/A')}")
                if stationarity.get("critical_values"):
                    logger.info(f"     • Critical Values:")
                    for level, value in stationarity.get("critical_values", {}).items():
                        logger.info(f"        - {level}: {value}")
                if stationarity.get("error"):
                    logger.info(f"     • Error: {stationarity.get('error')}")
            else:
                logger.info(f"\n  ⚠️  STATIONARITY: Insufficient data or not analyzed")
            
            # 6. Precipitation Extremes (if applicable)
            precipitation = analysis.get("precipitation", {})
            if precipitation:
                logger.info(f"\n  🌧️  PRECIPITATION EXTREMES:")
                logger.info(f"     • Zero Precipitation Ratio: {precipitation.get('zero_precipitation_ratio', 0):.3f}")
                logger.info(f"     • Extreme Precipitation Ratio: {precipitation.get('extreme_precipitation_ratio', 0):.3f}")
                logger.info(f"     • Range Impact: {precipitation.get('range_impact', 0):.3f}")
                logger.info(f"     • Max Precipitation: {precipitation.get('max_precipitation', 'N/A')} mm")
            
            # 7. Holt-Winters Penalty Breakdown
            hw_uncovered = param_data.get("holt_winters_uncovered", {})
            if hw_uncovered:
                logger.info(f"\n  🔻 HOLT-WINTERS PENALTY BREAKDOWN:")
                total_penalty = sum(hw_uncovered.values())
                logger.info(f"     • Total Penalty: {total_penalty:.2f}%")
                
                # Sort penalties by magnitude for better readability
                for reason, penalty in sorted(hw_uncovered.items(), key=lambda x: x[1], reverse=True):
                    # Better formatting for trend preservation
                    if reason == "trend_preservation_loss":
                        display_name = "Trend Preservation Loss"
                    else:
                        display_name = reason.replace('_', ' ').title()
                    
                    logger.info(f"     • {display_name}: -{penalty:.2f}%")
            else:
                logger.info(f"\n  ✅ HOLT-WINTERS: No penalties applied")

            # 8. LSTM Penalty Breakdown
            lstm_uncovered = param_data.get("lstm_uncovered", {})
            if lstm_uncovered:
                logger.info(f"\n  🔻 LSTM PENALTY BREAKDOWN:")
                total_penalty = sum(lstm_uncovered.values())
                logger.info(f"     • Total Penalty: {total_penalty:.2f}%")
                
                for reason, penalty in sorted(lstm_uncovered.items(), key=lambda x: x[1], reverse=True):
                    # Better formatting
                    if reason == "trend_preservation_loss":
                        display_name = "Trend Preservation Loss"
                    else:
                        display_name = reason.replace('_', ' ').title()
                    
                    logger.info(f"     • {display_name}: -{penalty:.2f}%")
            else:
                logger.info(f"\n  ✅ LSTM: No penalties applied")
            
            # 9. Data Quality Indicators
            logger.info(f"\n  📋 DATA QUALITY INDICATORS:")
            logger.info(f"     • Data Points: {analysis.get('data_points', len(processed_df))}")
            logger.info(f"     • Missing Ratio: {analysis.get('missing_ratio', 0)*100:.2f}%")
            if analysis.get("insufficient_data"):
                logger.info(f"     ⚠️  WARNING: Insufficient data for full analysis")
        
        # Global Uncovered Breakdown
        logger.info("\n" + "="*80)
        logger.info("GLOBAL PENALTY AGGREGATION:")
        logger.info("="*80)
        
        hw_global = coverage_data.get("holt_winters", {}).get("uncovered_breakdown", {})
        lstm_global = coverage_data.get("lstm", {}).get("uncovered_breakdown", {})
        
        if hw_global:
            logger.info(f"\n  🔻 Holt-Winters Average Penalties Across Parameters:")
            for reason, penalty in sorted(hw_global.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"     • {reason.replace('_', ' ').title()}: -{penalty:.2f}%")
        
        if lstm_global:
            logger.info(f"\n  🔻 LSTM Average Penalties Across Parameters:")
            for reason, penalty in sorted(lstm_global.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"     • {reason.replace('_', ' ').title()}: -{penalty:.2f}%")
        
        # Warnings Summary
        warnings = self.preprocessing_report.get("warnings", [])
        if warnings:
            logger.info("\n" + "="*80)
            logger.info("⚠️  WARNINGS SUMMARY:")
            logger.info("="*80)
            for i, warning in enumerate(warnings, 1):
                logger.info(f"  {i}. {warning}")
        
        logger.info("\n" + "="*80)
        logger.info("END OF DETAILED COVERAGE ANALYSIS")
        logger.info("="*80 + "\n")
    
    def _calculate_trend_penalty(self, trend_preservation: float) -> float:
        """
        Calculate penalty based on trend preservation loss
        
        Args:
            trend_preservation: Trend agreement ratio (0.0 to 1.0)
        
        Returns:
            Penalty percentage (0.0 to 20.0)
        
        Rationale:
        - Trend preservation measures how well smoothing maintains directional patterns
        - Low preservation means forecasting models will learn from distorted patterns
        - Critical for both Holt-Winters (trend component) and LSTM (sequential learning)
        
        Thresholds:
        - ≥90%: Excellent preservation, no penalty
        - 85-90%: Minor loss, models still reliable (-3%)
        - 80-85%: Moderate loss, noticeable forecast degradation (-7%)
        - 75-80%: Significant loss, high forecast uncertainty (-12%)
        - <75%: Severe loss, forecasting unreliable (-20%)
        """
        if trend_preservation >= 0.90:
            return 0.0
        elif trend_preservation >= 0.85:
            return 3.0
        elif trend_preservation >= 0.80:
            return 7.0
        elif trend_preservation >= 0.75:
            return 12.0
        else:
            return 20.0
    
    def _calculate_seasonality_penalty(
        self, 
        seasonal_strength: float
    ) -> float:
        """
         Calculate gradient-based seasonality penalty for Holt-Winters
    
        Args:
            seasonal_strength: Seasonal variance ratio (0.0 to 1.0)
        
        Returns:
            Penalty percentage (0.0 to 25.0)
        
        Rationale:
        - Holt-Winters Seasonal method requires clear seasonal patterns
        - Weak/absent seasonality causes poor seasonal component estimation
        - Model may overfit noise or fail to capture true patterns
        
        Thresholds (based on seasonal decomposition literature):
        - <0.1: No seasonality, HW inappropriate (-25%)
        - 0.1-0.2: Very weak, HW struggles (-20%)
        - 0.2-0.3: Weak but usable with caution (-12%)
        - 0.3-0.5: Acceptable, not optimal (-5%)
        - ≥0.5: Strong seasonality, HW ideal (0%)
        
        Note: LSTM is less affected by seasonality, so this penalty
        is applied with reduced weight for LSTM coverage calculation
        """
        if seasonal_strength < 0.1:
            return 25.0
        elif seasonal_strength < 0.2:
            return 20.0
        elif seasonal_strength < 0.3:
            return 12.0
        elif seasonal_strength < 0.5:
            return 5.0
        else:
            return 0.0
    
    def _get_smoothing_quality(self, param: str) -> Dict[str, Any]:
        """Get smoothing quality metrics from validation results"""
        smoothing_validation = self.preprocessing_report.get("smoothing_validation", {})
        
        if param not in smoothing_validation:
            return {"penalty": 0.0, "trend_value": 0.0}
        
        validation_data = smoothing_validation[param]
        gcv_score = validation_data.get("gcv_score", 0.0)
        trend_pct = validation_data.get("trend_preservation_pct", 100.0) / 100.0
        
        # Convert GCV to penalty
        if gcv_score < 2.0:
            gcv_penalty = 0.0
        elif gcv_score < 4.0:
            gcv_penalty = 5.0
        elif gcv_score < 10.0:
            gcv_penalty = 10.0
        else:
            gcv_penalty = 20.0
        
        return {
            "penalty": gcv_penalty,
            "trend_value": trend_pct
        }
        
    def _optimize_parameter_alphas(self, df: pd.DataFrame) -> None:
        logger.info("🔍 Starting adaptive alpha optimization...")
        
        params = self.options.get("columns_to_process", [])
        alpha_range = self.options.get("alpha_optimization_range", [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20])
        
        optimization_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            param_config = self.options.get("parameter_configs", {}).get(param, {})

            # Determine smoothing method with proper fallback chain
            smoothing_method = param_config.get(
                "smoothing_method",
                self.options.get("smoothing_method", "exponential")
            )

            # Skip if no smoothing or non-exponential
            if smoothing_method is None:
                logger.info(f"  ⏭️  {param}: Skipping (no smoothing configured)")
                continue
                
            if smoothing_method != "exponential":
                logger.info(f"  ⏭️  {param}: Skipping (using {smoothing_method} method)")
                continue
            
            optimal_alpha = self._find_optimal_alpha_for_parameter(df, param, alpha_range)

            # Store in config - ensure parameter_configs exists
            if "parameter_configs" not in self.options:
                self.options["parameter_configs"] = {}

            # Initialize parameter config if not exists
            if param not in self.options["parameter_configs"]:
                self.options["parameter_configs"][param] = {}

            # Update alpha
            self.options["parameter_configs"][param]["exponential_alpha"] = optimal_alpha
            optimization_results[param] = optimal_alpha

            logger.info(f"Optimal α={optimal_alpha:.3f} for {param}")

            # Store results in report
            self.preprocessing_report["alpha_optimization"] = optimization_results
            logger.info(f"Alpha optimization complete: {len(optimization_results)} parameters optimized")
        
        # Store results
        self.preprocessing_report["alpha_optimization"] = optimization_results
        logger.info(f"Alpha optimization complete")
        
    def _find_optimal_alpha_for_parameter(
        self,
        df: pd.DataFrame,
        param: str,
        alpha_range: List[float]
    ) -> float:
        """
        Test multiple alpha values and return the optimal one
        Scoring: (trend_preservation * 100) - (GCV * 0.5)
        """
        
        if param not in df.columns:
            logger.warning(f"Parameter {param} not in dataframe - using default alpha")
            return self.options.get("exponential_alpha", 0.15)
        
        series = df[param].dropna()
        
        # Check minimum data requirement (lowered threshold for better coverage)
        if len(series) < 100:  # Need at least 100 points for meaningful optimization
            logger.warning(f"Parameter {param}: insufficient data ({len(series)} points < 100) - using default alpha")
            return self.options.get("exponential_alpha", 0.15)
        
        # Initialize with global default
        best_alpha = self.options.get("exponential_alpha", 0.15)
        best_score = -np.inf
        test_results = []
        
        logger.info(f"     Testing {len(alpha_range)} alpha values...")
        
        for alpha in alpha_range:
            performance = self._test_alpha_performance(series.values, alpha)
            
            if performance is None:
                continue
            
            gcv_score = performance.get("gcv_score", 10.0)
            trend_pct = performance.get("trend_preservation_pct", 0.0)
            
            # Scoring: prioritize trend preservation
            combined_score = (trend_pct * 1.0) - (gcv_score * 0.5)
            
            test_results.append({
                "alpha": alpha,
                "gcv": gcv_score,
                "trend": trend_pct,
                "score": combined_score
            })
            
            logger.info(f"     α={alpha}: GCV={gcv_score:.4f}, Trend={trend_pct:.1f}%, Score={combined_score:.2f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_alpha = alpha
        
        if test_results:
            best_result = max(test_results, key=lambda x: x["score"])
            logger.info(f"     🏆 BEST: α={best_result['alpha']}, Trend={best_result['trend']:.1f}%")
        
        return best_alpha

    def _test_alpha_performance(
        self,
        original_series: np.ndarray,
        alpha: float
    ) -> Optional[Dict[str, Any]]:
        """
        Test a single alpha value and return GCV + trend preservation metrics
        """
        
        try:
            # Validate input size
            if len(original_series) < 100:  # Match minimum from _find_optimal_alpha_for_parameter
                return None
            
            # Check for NaN issues - both all NaN and too many NaN
            nan_count = np.isnan(original_series).sum()
            if nan_count == len(original_series):
                logger.warning(f"All-NaN data detected for α={alpha}")
                return None
            
            if nan_count > len(original_series) * 0.5:  # More than 50% NaN
                logger.warning(f"Too many NaN values ({nan_count}/{len(original_series)}) for α={alpha}")
                return None

            # Validate alpha range (should already be validated, but defensive)
            if alpha <= 0 or alpha >= 1:
                logger.warning(f"Invalid alpha value: {alpha} (must be between 0 and 1)")
                return None
            
            # Apply exponential smoothing on cleaned series
            series_pd = pd.Series(original_series).dropna()  # Drop NaN before smoothing
            
            if len(series_pd) < 100:  # Check again after dropping NaN
                logger.warning(f"Insufficient valid data after dropping NaN for α={alpha}")
                return None
                
            smoothed_series = series_pd.ewm(alpha=alpha, adjust=False).mean().values
            original_clean = original_series[~np.isnan(original_series)]
            
            # Ensure same length
            min_len = min(len(original_clean), len(smoothed_series))
            original_clean = original_clean[:min_len]
            smoothed_series = smoothed_series[:min_len]
            
            gcv_score = self._calculate_gcv(original_clean, smoothed_series, param=None)
            trend_preservation = self._calculate_trend_preservation(original_clean, smoothed_series)
            
            return {
                "alpha": alpha,
                "gcv_score": gcv_score,
                "trend_preservation_pct": trend_preservation * 100,
                "combined_score": (trend_preservation * 100) - (gcv_score * 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Error testing α={alpha}: {str(e)}")
            return None
        
    def _calculate_holt_winters_coverage(
        self, 
        large_gaps, 
        extreme_outliers, 
        seasonality, 
        stationarity, 
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate Holt-Winters model coverage with refined penalties
        
        Phase 1 Improvements:
        1. Added independent trend preservation penalty
        2. Gradient-based seasonality penalty
        3. Integrated GCV smoothing quality
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality
        smoothing_quality = self._get_smoothing_quality(param)
        gcv_penalty = smoothing_quality["penalty"]
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation ( Phase 1)
        trend_value = smoothing_quality.get("trend_value", 0.0)
        
        if trend_value > 0:  # Only if smoothing was applied
            trend_penalty = self._calculate_trend_penalty(trend_value)
            
            if trend_penalty > 0:
                base_coverage -= trend_penalty
                uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
                
                # Log warning for severe trend loss
                if trend_penalty >= 12.0:
                    logger.warning(
                        f"Parameter {param}: Severe trend preservation loss "
                        f"({trend_value*100:.1f}%) - High forecast uncertainty"
                    )
        
        # PENALTY 3: Large Gaps
        large_gap_penalty = large_gaps.get("impact_percentage", 0)
        
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 4: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 5: Weak Seasonality (REFINED - Phase 1)
        seasonal_strength = seasonality.get("seasonal_strength", 0)
        has_clear_seasonality = seasonality.get("has_clear_seasonality", False)
        
        # Use refined gradient-based penalty
        seasonal_penalty = self._calculate_seasonality_penalty(seasonal_strength)
        
        if seasonal_penalty > 0:
            base_coverage -= seasonal_penalty
            uncovered_reasons["weak_seasonality"] = round(seasonal_penalty, 2)
            
            # Log info about seasonality quality
            if seasonal_penalty >= 20.0:
                logger.warning(
                    f"Parameter {param}: Very weak seasonality "
                    f"(strength={seasonal_strength:.3f}) - Holt-Winters may be inappropriate"
                )
            elif seasonal_penalty >= 12.0:
                logger.info(
                    f"Parameter {param}: Weak seasonality "
                    f"(strength={seasonal_strength:.3f}) - Holt-Winters suboptimal"
                )
        
        # PENALTY 6: Non-Stationarity
        if not stationarity.get("is_stationary", False):
            stationarity_penalty = 5.0
            base_coverage -= stationarity_penalty
            uncovered_reasons["non_stationary"] = round(stationarity_penalty, 2)
        
        # PENALTY 7: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        # PENALTY 8: Compound Penalty (if multiple serious issues)
        issue_count = 0
        
        if large_gap_penalty > 5:
            issue_count += 1
        if outlier_penalty > 3:
            issue_count += 1
        if seasonal_penalty >= 12:  # refined threshold
            issue_count += 1
        if gcv_penalty > 10:
            issue_count += 1
        if trend_penalty >= 12:  #  count trend as serious issue
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(
                f"Parameter {param}: {issue_count} serious issues detected "
                f"- Compound penalty applied (forecasting highly uncertain)"
            )
        
        # FINAL COVERAGE
        final_coverage = max(0, base_coverage)
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
    
    
    def _calculate_lstm_coverage(
        self, 
        large_gaps, 
        extreme_outliers, 
        precipitation, 
        stationarity, 
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate LSTM model coverage with refined penalties
        
        Phase 1 Improvements:
        1. Added trend preservation penalty (slightly reduced weight vs HW)
        2. Integrated GCV smoothing quality
        
        Note: LSTM is less sensitive to seasonality than Holt-Winters,
        so seasonality penalty is not applied here
        """
        
        base_coverage = 100.0
        uncovered_reasons = {}
        
        # PENALTY 1: GCV Smoothing Quality (80% weight vs HW)
        smoothing_quality = self._get_smoothing_quality(param)
        gcv_penalty = smoothing_quality["penalty"] * 0.8  # Slightly less critical than HW
        
        if gcv_penalty > 0:
            base_coverage -= gcv_penalty
            uncovered_reasons["smoothing_quality"] = round(gcv_penalty, 2)
        
        # PENALTY 2: Trend Preservation (Phase 1, 80% weight)
        trend_value = smoothing_quality.get("trend_value", 0.0)
        
        if trend_value > 0:  # Only if smoothing was applied
            trend_penalty_full = self._calculate_trend_penalty(trend_value)
            trend_penalty = trend_penalty_full * 0.8  # 80% weight for LSTM
            
            if trend_penalty > 0:
                base_coverage -= trend_penalty
                uncovered_reasons["trend_preservation_loss"] = round(trend_penalty, 2)
        
        # PENALTY 3: Large Gaps (80% weight vs HW)
        large_gap_penalty = large_gaps.get("impact_percentage", 0) * 0.8
        
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 4: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 5: Precipitation Extremes (if applicable)
        precip_penalty = 0
        
        if precipitation:
            precip_penalty = precipitation.get("range_impact", 0) * 10
            
            if precip_penalty > 0:
                base_coverage -= precip_penalty
                uncovered_reasons["precipitation_extremes"] = round(precip_penalty, 2)
        
        # PENALTY 6: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        # PENALTY 7: Compound Penalty (if multiple serious issues)
        issue_count = 0
        
        if large_gap_penalty > 4:  # Slightly lower threshold than HW
            issue_count += 1
        if outlier_penalty > 3:
            issue_count += 1
        if precip_penalty > 5:
            issue_count += 1
        if gcv_penalty > 8:
            issue_count += 1
        if trend_penalty >= 10:  #  count trend as serious issue (80% of 12)
            issue_count += 1
        
        if issue_count >= 3:
            compound_penalty = 15.0
            base_coverage -= compound_penalty
            uncovered_reasons["compound_issues"] = round(compound_penalty, 2)
            
            logger.warning(
                f"Parameter {param}: {issue_count} serious issues detected "
                f"- Compound penalty applied (LSTM forecasting uncertain)"
            )
        
        # FINAL COVERAGE
        final_coverage = max(0, base_coverage)
        
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
                df[param] = self._smooth_exponential(df[param], param)  # Pass param
                smoothing_summary[param] = "exponential"
        
        self.preprocessing_report["smoothing"] = {
            "method": smoothing_method,
            "parameters_smoothed": smoothing_summary
        }
        
        logger.info(f"Smoothing applied: {smoothing_summary}")
        
        return df
    
    def _validate_smoothing_method(self, df: pd.DataFrame) -> None:
        """
        Validate smoothing quality using GCV + Trend Preservation
        """
        
        logger.info("Validating smoothing quality...")
        
        params = self.options.get("columns_to_process", [])
        validation_results = {}
        
        for param in params:
            if param not in df.columns:
                continue
            
            # Check smoothing_summary instead of config
            smoothing_summary = self.preprocessing_report.get("smoothing", {}).get("parameters_smoothed", {})
            param_smoothing_applied = smoothing_summary.get(param, "none")
            
            # Skip parameters where smoothing was NOT applied
            if param_smoothing_applied == "none":
                validation_results[param] = {
                    "status": "no_smoothing",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            
            # Get original and smoothed values
            if self.original_data is None or param not in self.original_data.columns:
                logger.warning(f"Original data not available for {param}")
                validation_results[param] = {
                    "status": "missing_original_data",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue        
            
            original = self.original_data.reset_index()[['Date', param]].set_index('Date')[param].dropna()
            smoothed = df.reset_index()[['Date', param]].set_index('Date')[param].dropna() if 'Date' in df.columns else df[param].dropna()
            common_idx = original.index.intersection(smoothed.index)
            if len(common_idx) == 0:
                logger.warning(f"No common indices found between original and smoothed data for {param}")
                validation_results[param] = {
                    "status": "no_common_indices",
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            original_aligned = original.loc[common_idx]
            smoothed_aligned = smoothed.loc[common_idx]
            
            # Check if enough data
            if len(common_idx) < 30:
                validation_results[param] = {
                    "status": "insufficient_data",
                    "data_points": len(common_idx),
                    "gcv": None,
                    "trend_preservation": None
                }
                continue
            
            # 1. Calculate GCV 
            gcv_score = self._calculate_gcv(
                original_aligned.values,
                smoothed_aligned.values,
                param
            )
            
            # 2. Calculate Trend Preservation
            trend_agreement = self._calculate_trend_preservation(
                original_aligned.values,
                smoothed_aligned.values
            )
            
            # 3. Determine quality 
            quality_status = self._determine_smoothing_quality(
                gcv_score,
                trend_agreement
            )
            
            validation_results[param] = {
                "gcv_score": round(float(gcv_score), 4),
                "trend_preservation_pct": round(float(trend_agreement * 100), 2),
                "quality_status": quality_status,
                "smoothing_method": param_smoothing_applied,
                "data_points": len(common_idx)
            }
            
            # Log individual parameter results
            logger.info(
                f"Parameter {param}: "
                f"GCV={gcv_score:.4f}, "
                f"Trend={trend_agreement*100:.2f}%, "
                f"Quality={quality_status.upper()}"
            )
            
            # Add warnings if quality is poor
            if quality_status == "poor":
                self.preprocessing_report["warnings"].append(
                    f"Parameter {param}: smoothing quality is poor "
                    f"(GCV={gcv_score:.3f}, Trend={trend_agreement*100:.1f}%)"
                )
        
        # Store in report
        self.preprocessing_report["smoothing_validation"] = validation_results
        logger.info(f"Smoothing validation completed for {len(validation_results)} parameters")
        
    def _determine_smoothing_quality(self, gcv: float, trend_preservation: float) -> str:
        """
        Determine overall smoothing quality based on GCV and trend preservation
        
        Args:
            gcv: Generalized Cross-Validation score (lower = better)
            trend_preservation: Trend agreement ratio (0.0 to 1.0, higher = better)
        
        Returns:
            Quality status: "excellent", "good", "fair", "poor"
        """
        if gcv < 2.0 and trend_preservation > 0.80:
            return "excellent"
        elif gcv < 4.0 and trend_preservation > 0.75:
            return "good"
        elif gcv < 10.0 and trend_preservation > 0.70:
            return "fair"
        else:
            return "poor"
        
    def _calculate_gcv(self, original: np.ndarray, smoothed: np.ndarray, param: str) -> float:
        """
        Calculate Generalized Cross-Validation score
        Lower GCV = better smoothing balance between fit and complexity
        
        GCV penalizes both:
        - Poor fit (high MSE)
        - Over-smoothing (high effective degrees of freedom)
        """
        n = len(original)
        
        # Calculate Mean Squared Error
        mse = np.mean((original - smoothed) ** 2)
        
        # Get parameter-specific smoothing method (not global!)
        param_config = self.options.get("parameter_configs", {}).get(param, {})
        smoothing_method = param_config.get(
            "smoothing_method", 
            self.options.get("smoothing_method", "exponential")  # Fallback to global
        )
        
        # Handle case where smoothing_method is None (defensive programming)
        if smoothing_method is None:
            logger.warning(f"GCV called for parameter {param} with no smoothing method")
            return 0.0  # Return 0 to indicate no smoothing was applied
        
                # Estimate effective degrees of freedom based on smoothing method
        if smoothing_method == "moving_average":
            window_size = self.options.get("window_size", 5)
            edf = window_size
            
        elif smoothing_method == "exponential":
            # Get parameter-specific alpha correctly
            if param:
                param_config = self.options.get("parameter_configs", {}).get(param, {})
                alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
            else:
                alpha = self.options.get("exponential_alpha", 0.15)
            
            # Validate alpha range
            if alpha <= 0:
                alpha = 0.01  # Minimum alpha
                logger.warning(f"Alpha too low, using minimum: {alpha}")
            elif alpha >= 1:
                alpha = 0.99  # Maximum alpha
                logger.warning(f"Alpha too high, using maximum: {alpha}")
            
            # CORRECTED FORMULA for Exponential Weighted Moving Average (EWMA)
            # Reference: Hyndman & Athanasopoulos "Forecasting: Principles and Practice"
            # For EWMA, effective degrees of freedom: edf ≈ 1 / (2 - alpha)
            # Alternative simple approximation: edf ≈ 1/alpha (for small alpha)
            
            # Using the more accurate formula:
            edf = 1.0 / (2.0 - alpha)
            
            # Ensure reasonable bounds
            edf = max(1.0, min(edf, n / 3.0))  # EDF should be between 1 and n/3
            
            # Debug logging for verification
            # logger.debug(f"GCV calculation: alpha={alpha:.3f}, edf={edf:.2f}, n={n}")
            
        else:
            edf = 5  # Default conservative estimate
            if smoothing_method:
                logger.warning(f"Unknown smoothing method '{smoothing_method}' for parameter {param}, using default EDF=5")

        # GCV formula: MSE / (1 - edf/n)²
        # Protection against numerical issues
        denominator = (1.0 - edf / n) ** 2

        # Ensure denominator is not too small (prevents division explosion)
        if denominator <= 0.01:
            logger.warning(f"GCV denominator too small ({denominator:.4f}), clamping to 0.01")
            denominator = 0.01

        gcv = mse / denominator

        return gcv
    
    def _calculate_non_smoothed_coverage(
        self,
        large_gaps,
        extreme_outliers,
        coverage_analysis,
        param: str
    ) -> Dict[str, Any]:
        """
        Calculate coverage for parameters that don't need smoothing
        
        Args:
            large_gaps: Large gaps analysis
            extreme_outliers: Extreme outliers analysis  
            coverage_analysis: Basic coverage data
            param: Parameter name
            
        Returns:
            Coverage dictionary for non-smoothed parameters
        """
        base_coverage = 95.0  # High base coverage - no smoothing needed
        uncovered_reasons = {}
        
        # PENALTY 1: Large Gaps
        large_gap_penalty = large_gaps.get("impact_percentage", 0)
        if large_gap_penalty > 0:
            base_coverage -= large_gap_penalty
            uncovered_reasons["large_gaps"] = round(large_gap_penalty, 2)
        
        # PENALTY 2: Extreme Outliers
        outlier_penalty = extreme_outliers.get("impact_percentage", 0)
        if outlier_penalty > 0:
            base_coverage -= outlier_penalty
            uncovered_reasons["extreme_outliers"] = round(outlier_penalty, 2)
        
        # PENALTY 3: Missing Data
        missing_penalty = coverage_analysis.get("missing_ratio", 0) * 25
        if missing_penalty > 0:
            base_coverage -= missing_penalty
            uncovered_reasons["missing_data"] = round(missing_penalty, 2)
        
        final_coverage = max(0, base_coverage)
        
        return {
            "coverage_percentage": round(final_coverage, 2),
            "uncovered_reasons": uncovered_reasons
        }
    
    def _calculate_trend_preservation(self, original: np.ndarray, smoothed: np.ndarray) -> float:
        """
        Calculate trend direction agreement between original and smoothed data
        
        Returns: percentage of matching trend directions (0.0 to 1.0)
        
        Special cases:
        - Returns 1.0 if data is flat (no trends to preserve = perfect preservation)
        - Returns np.nan if insufficient data for meaningful calculation
        """
        # Validate input lengths
        if len(original) != len(smoothed):
            logger.warning(f"Trend preservation: length mismatch (original={len(original)}, smoothed={len(smoothed)})")
            return np.nan
        
        if len(original) < 2:  # Need at least 2 points for diff
            logger.warning(f"Trend preservation: insufficient data (n={len(original)})")
            return np.nan
        
        # Calculate first differences (trend direction)
        original_diff = np.diff(original)
        smoothed_diff = np.diff(smoothed)
        
        # Get signs (direction: +1 for increase, -1 for decrease, 0 for no change)
        original_direction = np.sign(original_diff)
        smoothed_direction = np.sign(smoothed_diff)
        
        # Calculate agreement (exclude zeros - flat regions)
        non_zero_mask = (original_direction != 0) & (smoothed_direction != 0)
        
        if non_zero_mask.sum() == 0:
            # No clear trends in data (all flat) - this is GOOD, not bad!
            # Flat data means smoothing preserved the flat nature perfectly
            logger.debug(f"Trend preservation: No clear trends detected (flat data) - returning 1.0 (perfect)")
            return 1.0  # ← FIXED: was 0.0
        
        if non_zero_mask.sum() < 10:  # Too few trend changes for reliable calculation
            logger.debug(f"Trend preservation: Very few trend changes ({non_zero_mask.sum()}) - may be unreliable")
            # Still calculate but warn
        
        # Check where directions match
        agreement = (original_direction[non_zero_mask] == smoothed_direction[non_zero_mask])
        trend_preservation = agreement.mean()
        
        return trend_preservation
    
    def _smooth_moving_average(self, series: pd.Series) -> pd.Series:
        """Apply moving average smoothing"""
        window_size = self.options.get("window_size", 5)
        return series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    def _smooth_exponential(
        self,
        series: pd.Series,
        param: str = None
    ) -> pd.Series:
        """Apply exponential smoothing with parameter-specific alpha"""
        
        if param:
            param_config = self.options.get("parameter_configs", {}).get(param, {})
            alpha = param_config.get("exponential_alpha", self.options.get("exponential_alpha", 0.15))
        else:
            alpha = self.options.get("exponential_alpha", 0.15)
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