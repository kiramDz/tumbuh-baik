from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
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
                return{
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
            missing_fields = [fields for fields in field_requirements if fields not in first_doc]

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
                field_stats[field] = {  # ✅ Calculated once after loop
                    'missing_count': missing_values,
                    'missing_percentage': round(missing_pct, 2),
                    'invalid_types': invalid_types,
                    'out_of_range': out_of_range,  # ✅ Correct variable
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

                if out_of_range > sample_size * 0.1:  # More than 10% out of range
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
                    f"Dataset has only {total_records} records."
                    " ML models may not perform well with limited data."
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
            
            # ✅ Fix: Remove redundant date field consistency check
            # The original code checked for Year, Month, Day fields but preprocessing doesn't require them
            
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

            # Generate new _id values instead of keeping the original ones
            if '_id' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('_id', axis=1)
                
            # ✅ TAMBAHKAN INI: Drop temporary preprocessing columns
            temp_columns = ['Season', 'is_RR_missing']
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
        """Update dataset-meta after preprocessing"""
        try:
            # Find the metadata collection
            meta_collection = None
            if "dataset_meta" in db.list_collection_names():
                meta_collection = "dataset_meta"
            elif "DatasetMeta" in db.list_collection_names():
                meta_collection = "DatasetMeta"
            else:
                logger.warning("No metadata collection found, skipping metadata update")
                return {"status": "no_metadata_collection"}

            # Get the original dataset metadata
            original_meta = db[meta_collection].find_one({"collectionName": original_collection_name})

            if not original_meta:
                logger.warning(f"No metadata found for collection '{original_collection_name}', skipping metadata update")
                return {"status": "no_original_metadata"}

            # Get field list from first document
            sample_doc = db[cleaned_collection_name].find_one()
            columns = list(sample_doc.keys()) if sample_doc else []

            # Create new metadata for cleaned collection
            cleaned_meta = {
                "name": f"{original_meta.get('name', original_collection_name)} (Cleaned)",
                "source": original_meta.get('source'),
                "filename": f"{original_meta.get('filename')}",
                "collectionName": cleaned_collection_name,
                "fileSize": record_count * 250,  # Rough estimate
                "totalRecords": record_count,
                "fileType": "json",
                "status": "preprocessed",
                "columns": columns,
                "description": original_meta.get('description'),
                "uploadDate": datetime.now(),
                "isAPI": original_meta.get('isAPI', False),
                "lastUpdated": datetime.now(),
                "preprocessedFrom": original_collection_name,
                "apiConfig": original_meta.get('apiConfig', {})
            }

            # Update metadata for original dataset
            db[meta_collection].update_one(
                {"collectionName": original_collection_name},
                {"$set": {"status": "preprocessed"}}
            )

            # Check if cleaned metadata already exists
            existing_cleaned = db[meta_collection].find_one({"collectionName": cleaned_collection_name})

            if existing_cleaned:
                # Update existing metadata
                db[meta_collection].update_one(
                    {"collectionName": cleaned_collection_name},
                    {"$set": cleaned_meta}
                )
                logger.info(f"Updated metadata for '{cleaned_collection_name}'")
            else:
                # Insert new metadata
                db[meta_collection].insert_one(cleaned_meta)
                logger.info(f"Created new metadata for '{cleaned_collection_name}'")

            return {
                "status": "success",
                "originalStatus": "preprocessed",
                "cleanedCollectionName": cleaned_collection_name
            }

        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
class BmkgPreprocessor:
    """Main class for preprocessing BMKG datasets"""

    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.validator = BmkgDataValidator()
        self.loader = BmkgDataLoader()
        self.saver = BmkgDataSaver()

        # Track preprocessing statistics
        self.preprocessing_stats = {
            'outliers_removed': {},
            'values_imputed': {},
            'imputation_methods': {},
            'model_performance': {}
        }

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing to BMKG data
        Following the robust imputation strategies
        """
        logger.info("=" * 60)
        logger.info("Starting BMKG Data Preprocessing")
        logger.info(f"Initial dataset shape: {df.shape}")
        logger.info("=" * 60)

        # Make a copy to avoid modifying original
        processed_df = df.copy()

        # Ensure Date is datetime and set as index
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'])
            processed_df = processed_df.set_index('Date').sort_index()

        # Add temporal features
        processed_df['month'] = processed_df.index.month
        processed_df['Season'] = processed_df.index.month.map(
            lambda m: 'Wet' if m in [9, 10, 11, 12, 1, 2, 3] else 'Dry'
        )

        # Step 1: Replace missing value codes with NaN
        processed_df = self._handle_missing_codes(processed_df)

        # Step 2: Detect and remove outliers
        processed_df = self._detect_and_handle_outliers(processed_df)

        # Step 3: Impute missing values for each variable
        # Order matters! Temperature and humidity first (needed for rainfall model)
        processed_df = self._impute_temperature(processed_df)
        processed_df = self._impute_humidity(processed_df)
        processed_df = self._impute_rainfall(processed_df)
        processed_df = self._impute_wind_speed_max(processed_df)
        processed_df = self._impute_wind_speed_avg(processed_df)
        processed_df = self._impute_wind_direction_degrees(processed_df)
        processed_df = self._impute_sunshine_duration(processed_df)
        processed_df = self._impute_wind_direction_cardinal(processed_df)

        # Step 4: Apply physical constraints
        processed_df = self._apply_physical_constraints(processed_df)

        # Step 5: Final validation
        validation_result = self._validate_preprocessed_data(processed_df)

        if not validation_result['valid']:
            logger.warning("Post-preprocessing validation issues detected:")
            for issue in validation_result['issues']:
                logger.warning(f"  - {issue}")

        # Reset index to have Date as column
        processed_df = processed_df.reset_index()

        logger.info("=" * 60)
        logger.info("Preprocessing completed successfully")
        logger.info(f"Final dataset shape: {processed_df.shape}")
        logger.info("=" * 60)

        return processed_df

    def _handle_missing_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace missing value codes (8888, 9999) with NaN"""
        logger.info("Step 1: Handling missing value codes...")

        missing_codes = [8888.0, 9999.0, 8888, 9999]
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        total_replaced = 0
        for col in numeric_columns:
            for code in missing_codes:
                mask = df[col] == code
                count = mask.sum()
                if count > 0:
                    df.loc[mask, col] = np.nan
                    total_replaced += count
                    logger.info(f"  Replaced {count} instances of {code} in '{col}'")

        logger.info(f"  Total missing codes replaced: {total_replaced}")
        return df

    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers based on valid ranges"""
        logger.info("Step 2: Detecting and handling outliers...")

        valid_ranges = {
            'TX': (-5, 45),
            'TN': (-10, 35),
            'TAVG': (-5, 40),
            'RH_AVG': (0, 100),
            'RR': (0, 500),
            'FF_X': (0, 50),
            'FF_AVG': (0, 30),
            'SS': (0, 14),
            'DDD_X': (0, 360)
        }

        for col, (min_val, max_val) in valid_ranges.items():
            if col not in df.columns:
                continue

            # Detect outliers
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            outlier_count = invalid_mask.sum()

            if outlier_count > 0:
                logger.info(f"  Found {outlier_count} outliers in '{col}'")
                df.loc[invalid_mask, col] = np.nan

                # Track statistics
                self.preprocessing_stats['outliers_removed'][col] = int(outlier_count)

        return df

    def _impute_rainfall(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute rainfall using probabilistic Random Forest approach
        """
        logger.info("Step 3c: Imputing Rainfall (RR) - Probabilistic Approach...")

        # Create flag for originally missing values
        df['is_RR_missing'] = df['RR'].isna().astype(int)
        missing_count = df['is_RR_missing'].sum()

        if missing_count == 0:
            logger.info("  No missing rainfall values detected")
            return df

        logger.info(f"  Found {missing_count} missing rainfall values")

        try:
            # Prepare predictors (ensure they're already imputed)
            predictors = ['TAVG', 'RH_AVG', 'SS', 'TX', 'TN', 'month']
            available_predictors = [p for p in predictors if p in df.columns]

            # Fill any remaining NaNs in predictors with forward/backward fill
            for col in available_predictors:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

            # Create training data (where RR is known)
            train_data = df.dropna(subset=['RR'] + available_predictors)

            if len(train_data) < 100:
                logger.warning("  Insufficient data for ML model, using simple interpolation")
                df['RR'] = df['RR'].interpolate(method='time', limit_direction='both')
                df['RR'] = df['RR'].clip(lower=0)
                self.preprocessing_stats['imputation_methods']['RR'] = 'interpolation_fallback'
                return df

            # Create binary target: rain or no rain
            train_data['rain_occurred'] = (train_data['RR'] > 0).astype(int)

            # Train Random Forest classifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score

            X = train_data[available_predictors]
            y = train_data['rain_occurred']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )

            logger.info("  Training rain occurrence model...")
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"  Model performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            # Store model performance
            self.preprocessing_stats['model_performance']['rainfall_classifier'] = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'features_used': available_predictors
            }

            # Predict for missing values
            missing_mask = df['RR'].isna()
            X_missing = df.loc[missing_mask, available_predictors].copy()

            # Fill any NaNs in predictors
            for col in X_missing.columns:
                if X_missing[col].isna().any():
                    X_missing[col] = X_missing[col].fillna(df[col].median())

            rain_predicted = clf.predict(X_missing)

            # Set dry days to 0
            missing_indices = df.index[missing_mask]
            dry_indices = missing_indices[rain_predicted == 0]
            df.loc[dry_indices, 'RR'] = 0.0

            dry_count = len(dry_indices)
            rainy_count = missing_count - dry_count

            logger.info(f"  Predicted {dry_count} dry days (RR=0)")
            logger.info(f"  Sampling values for {rainy_count} rainy days")

            # For rainy days, sample from historical distributions
            rainy_indices = missing_indices[rain_predicted == 1]

            # Build distributions by month
            monthly_distributions = {}
            for month in range(1, 13):
                month_rain = df[(df['month'] == month) & (df['RR'] > 0) & (~df['RR'].isna())]['RR']
                if len(month_rain) > 5:
                    monthly_distributions[month] = month_rain

            # Sample for each rainy day
            for idx in rainy_indices:
                month = idx.month if hasattr(idx, 'month') else df.loc[idx, 'Month']

                if month in monthly_distributions:
                    sampled_value = monthly_distributions[month].sample(1).values[0]
                else:
                    # Fallback to overall distribution
                    overall_rain = df[(df['RR'] > 0) & (~df['RR'].isna())]['RR']
                    sampled_value = overall_rain.sample(1).values[0] if len(overall_rain) > 0 else 0

                df.loc[idx, 'RR'] = sampled_value

            # Ensure no negative values
            df['RR'] = df['RR'].clip(lower=0)

            # Track statistics
            self.preprocessing_stats['values_imputed']['RR'] = int(missing_count)
            self.preprocessing_stats['imputation_methods']['RR'] = 'probabilistic_random_forest'

            logger.info(f"  ✓ Rainfall imputation completed")

        except Exception as e:
            logger.error(f"  Error in rainfall imputation: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to simple interpolation
            df['RR'] = df['RR'].interpolate(method='time', limit_direction='both')
            df['RR'] = df['RR'].clip(lower=0)
            self.preprocessing_stats['imputation_methods']['RR'] = 'interpolation_error_fallback'

        return df

    def _impute_wind_speed_avg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute average wind speed using Gradient Boosting with temporal features
        """
        logger.info("Step 3d: Imputing Average Wind Speed (FF_AVG)...")

        missing_count = df['FF_AVG'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing wind speed values detected")
            return df

        logger.info(f"  Found {missing_count} missing FF_AVG values")

        try:
            # Add temporal features (lag values for wind persistence)
            df['FF_AVG_lag1'] = df['FF_AVG'].shift(1)
            df['FF_AVG_lag7'] = df['FF_AVG'].shift(7)

            # Wind direction zones
            df['wind_zone'] = pd.cut(
                df['DDD_X'],
                bins=[0, 90, 180, 270, 360],
                labels=[0, 1, 2, 3]
            ).astype(float)

            # Prepare predictors
            predictors = ['TAVG', 'RH_AVG', 'FF_X', 'wind_zone',
                        'FF_AVG_lag1', 'FF_AVG_lag7', 'month']
            available_predictors = [p for p in predictors if p in df.columns]

            # Create training set
            train_mask = df['FF_AVG'].notna()
            for pred in available_predictors:
                train_mask = train_mask & df[pred].notna()

            if train_mask.sum() < 50:
                logger.warning("  Insufficient data for ML model, using interpolation")
                df['FF_AVG'] = df['FF_AVG'].interpolate(method='time', limit_direction='both')
                df.drop(['FF_AVG_lag1', 'FF_AVG_lag7', 'wind_zone'], axis=1, inplace=True, errors='ignore')
                self.preprocessing_stats['imputation_methods']['FF_AVG'] = 'interpolation_fallback'
                return df

            X_train = df.loc[train_mask, available_predictors]
            y_train = df.loc[train_mask, 'FF_AVG']

            # Train Gradient Boosting model
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.metrics import r2_score, mean_squared_error

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

            logger.info("  Training wind speed model...")
            model.fit(X_train, y_train)

            # Validate model
            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))

            logger.info(f"  Model performance: R²={r2:.4f}, RMSE={rmse:.4f}")

            # Store performance
            self.preprocessing_stats['model_performance']['wind_speed_model'] = {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'features_used': available_predictors
            }

            # Predict missing values
            missing_mask = df['FF_AVG'].isna()
            X_missing = df.loc[missing_mask, available_predictors].copy()

            # Fill NaNs in predictors
            for col in X_missing.columns:
                if X_missing[col].isna().any():
                    X_missing[col] = X_missing[col].fillna(df[col].median())

            predictions = model.predict(X_missing)
            predictions = np.clip(predictions, 0, 30)  # Physical limits

            df.loc[missing_mask, 'FF_AVG'] = predictions

            # Track statistics
            self.preprocessing_stats['values_imputed']['FF_AVG'] = int(missing_count)
            self.preprocessing_stats['imputation_methods']['FF_AVG'] = 'gradient_boosting'

            logger.info(f"  ✓ Wind speed imputation completed")

            # Cleanup temporary columns
            df.drop(['FF_AVG_lag1', 'FF_AVG_lag7', 'wind_zone'], axis=1, inplace=True, errors='ignore')

        except Exception as e:
            logger.error(f"  Error in wind speed imputation: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to interpolation
            df['FF_AVG'] = df['FF_AVG'].interpolate(method='time', limit_direction='both')
            df.drop(['FF_AVG_lag1', 'FF_AVG_lag7', 'wind_zone'], axis=1, inplace=True, errors='ignore')
            self.preprocessing_stats['imputation_methods']['FF_AVG'] = 'interpolation_error_fallback'

        return df

    def _apply_physical_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply physical constraints to ensure data consistency
        """
        logger.info("Step 5: Applying physical constraints...")

        # Temperature constraints: TX >= TAVG >= TN
        if all(col in df.columns for col in ['TX', 'TAVG', 'TN']):
            # Fix TAVG > TX
            violation_mask = df['TAVG'] > df['TX']
            if violation_mask.sum() > 0:
                logger.info(f"  Fixed {violation_mask.sum()} cases where TAVG > TX")
                df.loc[violation_mask, 'TAVG'] = df.loc[violation_mask, 'TX']

            # Fix TAVG < TN
            violation_mask = df['TAVG'] < df['TN']
            if violation_mask.sum() > 0:
                logger.info(f"  Fixed {violation_mask.sum()} cases where TAVG < TN")
                df.loc[violation_mask, 'TAVG'] = df.loc[violation_mask, 'TN']

        # Wind speed constraints: FF_X >= FF_AVG
        if 'FF_X' in df.columns and 'FF_AVG' in df.columns:
            violation_mask = df['FF_X'] < df['FF_AVG']
            if violation_mask.sum() > 0:
                logger.info(f"  Fixed {violation_mask.sum()} cases where FF_X < FF_AVG")
                df.loc[violation_mask, 'FF_X'] = df.loc[violation_mask, 'FF_AVG'] * 1.2

        # Humidity bounds
        if 'RH_AVG' in df.columns:
            df['RH_AVG'] = df['RH_AVG'].clip(0, 100)

        # Non-negative rainfall and sunshine
        if 'RR' in df.columns:
            df['RR'] = df['RR'].clip(lower=0)

        if 'SS' in df.columns:
            df['SS'] = df['SS'].clip(lower=0)

        logger.info("  ✓ Physical constraints applied")
        return df

    def _validate_preprocessed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate preprocessed data to ensure quality
        """
        logger.info("Step 6: Validating preprocessed data...")

        issues = []
        warnings = []

        # Check for remaining NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            for col, count in nan_counts.items():
                if count > 0:
                    issues.append(f"Column '{col}' still has {count} NaN values")

        # Check temperature relationships
        if all(col in df.columns for col in ['TX', 'TAVG', 'TN']):
            tx_tavg_violations = (df['TX'] < df['TAVG']).sum()
            tavg_tn_violations = (df['TAVG'] < df['TN']).sum()

            if tx_tavg_violations > 0:
                issues.append(f"Found {tx_tavg_violations} cases where TX < TAVG")
            if tavg_tn_violations > 0:
                issues.append(f"Found {tavg_tn_violations} cases where TAVG < TN")

        # Check wind speed relationships
        if 'FF_X' in df.columns and 'FF_AVG' in df.columns:
            violations = (df['FF_X'] < df['FF_AVG']).sum()
            if violations > 0:
                issues.append(f"Found {violations} cases where FF_X < FF_AVG")

        # Check for unrealistic values
        if 'RR' in df.columns:
            extreme_rain = (df['RR'] > 300).sum()
            if extreme_rain > 0:
                warnings.append(f"Found {extreme_rain} days with extreme rainfall (>300mm)")

        is_valid = len(issues) == 0

        logger.info(f"  Validation {'passed' if is_valid else 'failed'}")
        return{
            'valid': is_valid,
            'issues': issues,
            'warnings': warnings
        }

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute temperature variables maintaining physical relationships
        TX >= TAVG >= TN
        """
        logger.info("Step 3a: Imputing Temperature (TX, TN, TAVG)...")

        temp_vars = ['TX', 'TN', 'TAVG']
        missing_counts = {var: df[var].isna().sum() for var in temp_vars if var in df.columns}

        total_missing = sum(missing_counts.values())
        if total_missing == 0:
            logger.info("  No missing temperature values detected")
            return df

        logger.info(f"  Missing values: {missing_counts}")

        # Step 1: Calculate TAVG from TX and TN where both available
        if all(var in df.columns for var in temp_vars):
            calc_mask = df['TAVG'].isna() & df['TX'].notna() & df['TN'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TAVG'] = (df.loc[calc_mask, 'TX'] + df.loc[calc_mask, 'TN']) / 2
                logger.info(f"  Calculated {calc_mask.sum()} TAVG values from TX and TN")

        # Step 2: Estimate TX from TAVG and TN
        if 'TX' in df.columns and 'TAVG' in df.columns and 'TN' in df.columns:
            calc_mask = df['TX'].isna() & df['TAVG'].notna() & df['TN'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TX'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TN']
                logger.info(f"  Calculated {calc_mask.sum()} TX values from TAVG and TN")

        # Step 3: Estimate TN from TAVG and TX
        if 'TN' in df.columns and 'TAVG' in df.columns and 'TX' in df.columns:
            calc_mask = df['TN'].isna() & df['TAVG'].notna() & df['TX'].notna()
            if calc_mask.sum() > 0:
                df.loc[calc_mask, 'TN'] = 2 * df.loc[calc_mask, 'TAVG'] - df.loc[calc_mask, 'TX']
                logger.info(f"  Calculated {calc_mask.sum()} TN values from TAVG and TX")

        # Step 4: Interpolate remaining missing values
        for var in temp_vars:
            if var not in df.columns:
                continue

            if df[var].isna().any():
                before_count = df[var].isna().sum()
                df[var] = df[var].interpolate(method='time', limit_direction='both')
                after_count = df[var].isna().sum()

                interpolated = before_count - after_count
                if interpolated > 0:
                    logger.info(f"  Interpolated {interpolated} values for {var}")

                # Track statistics
                if var not in self.preprocessing_stats['values_imputed']:
                    self.preprocessing_stats['values_imputed'][var] = 0
                self.preprocessing_stats['values_imputed'][var] += int(before_count)
                self.preprocessing_stats['imputation_methods'][var] = 'physical_relationship_interpolation'

        logger.info("  ✓ Temperature imputation completed")
        return df

    def _impute_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute humidity using dewpoint temperature (more stable for interpolation)
        """
        logger.info("Step 3b: Imputing Humidity (RH_AVG)...")

        if 'RH_AVG' not in df.columns:
            logger.warning("  RH_AVG column not found")
            return df

        missing_count = df['RH_AVG'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing humidity values detected")
            return df

        logger.info(f"  Found {missing_count} missing RH_AVG values")

        # Magnus formula for dewpoint calculation
        def calculate_dewpoint(temp, rh):
            """Calculate dewpoint temperature using Magnus formula"""
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
            dewpoint = (b * alpha) / (a - alpha)
            return dewpoint

        def calculate_rh_from_dewpoint(temp, dewpoint):
            """Calculate RH from temperature and dewpoint"""
            a = 17.27
            b = 237.7
            rh = 100 * np.exp(
                (a * dewpoint / (b + dewpoint)) -
                (a * temp / (b + temp))
            )
            return np.clip(rh, 0, 100)

        # Calculate dewpoint for known values
        if 'TAVG' in df.columns:
            valid_mask = df['RH_AVG'].notna() & df['TAVG'].notna()
            if valid_mask.sum() > 0:
                df.loc[valid_mask, 'dewpoint_temp'] = calculate_dewpoint(
                    df.loc[valid_mask, 'TAVG'],
                    df.loc[valid_mask, 'RH_AVG']
                )

                # Interpolate dewpoint (more stable than RH)
                df['dewpoint_temp'] = df['dewpoint_temp'].interpolate(
                    method='time', limit_direction='both'
                )

                # Calculate RH from interpolated dewpoint
                missing_rh = df['RH_AVG'].isna() & df['TAVG'].notna() & df['dewpoint_temp'].notna()
                if missing_rh.sum() > 0:
                    df.loc[missing_rh, 'RH_AVG'] = calculate_rh_from_dewpoint(
                        df.loc[missing_rh, 'TAVG'],
                        df.loc[missing_rh, 'dewpoint_temp']
                    )
                    logger.info(f"  Calculated {missing_rh.sum()} RH values from dewpoint")

                # Cleanup temporary column
                df.drop('dewpoint_temp', axis=1, inplace=True, errors='ignore')

        # Adjust for rainfall (rainy days should have higher humidity)
        if 'RR' in df.columns:
            rainy_days = df['RR'] > 1
            low_rh_rainy = rainy_days & (df['RH_AVG'] < 70)

            if low_rh_rainy.sum() > 0:
                logger.info(f"  Adjusting {low_rh_rainy.sum()} rainy days with low RH")
                # Boost RH on rainy days (sample from 75-95%)
                df.loc[low_rh_rainy, 'RH_AVG'] = np.random.uniform(
                    75, 95, low_rh_rainy.sum()
                )

        # Final interpolation for any remaining NaNs
        if df['RH_AVG'].isna().any():
            df['RH_AVG'] = df['RH_AVG'].interpolate(method='time', limit_direction='both')

        # Apply bounds
        df['RH_AVG'] = df['RH_AVG'].clip(0, 100)

        # Track statistics
        self.preprocessing_stats['values_imputed']['RH_AVG'] = int(missing_count)
        self.preprocessing_stats['imputation_methods']['RH_AVG'] = 'dewpoint_interpolation'

        logger.info("  ✓ Humidity imputation completed")
        return df

    def _impute_wind_speed_max(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute maximum wind speed with constraint: FF_X >= FF_AVG
        Uses ratio method based on seasonal patterns
        """
        logger.info("Step 3c: Imputing Maximum Wind Speed (FF_X)...")

        if 'FF_X' not in df.columns:
            logger.warning("  FF_X column not found")
            return df

        missing_count = df['FF_X'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing FF_X values detected")
            return df

        logger.info(f"  Found {missing_count} missing FF_X values")

        # Calculate typical ratio FF_X / FF_AVG
        if 'FF_AVG' in df.columns:
            valid_mask = df['FF_X'].notna() & df['FF_AVG'].notna() & (df['FF_AVG'] > 0)

            if valid_mask.sum() > 10:
                df.loc[valid_mask, 'wind_ratio'] = df.loc[valid_mask, 'FF_X'] / df.loc[valid_mask, 'FF_AVG']

                # Calculate seasonal ratios (wind gusts vary by season)
                seasonal_ratios = df[valid_mask].groupby('Season')['wind_ratio'].agg(['mean', 'std'])

                logger.info(f"  Seasonal wind gust ratios:")
                for season in seasonal_ratios.index:
                    mean_ratio = seasonal_ratios.loc[season, 'mean']
                    logger.info(f"    {season}: {mean_ratio:.2f}x average speed")

                # Impute missing FF_X where FF_AVG exists
                missing_ffx = df['FF_X'].isna() & df['FF_AVG'].notna()

                if missing_ffx.sum() > 0:
                    for season in df['Season'].unique():
                        if pd.isna(season):
                            continue

                        season_mask = (df['Season'] == season) & missing_ffx

                        if season_mask.sum() > 0 and season in seasonal_ratios.index:
                            ratio_mean = seasonal_ratios.loc[season, 'mean']
                            ratio_std = seasonal_ratios.loc[season, 'std']

                            # Sample from normal distribution with reasonable bounds
                            sampled_ratios = np.random.normal(
                                ratio_mean,
                                ratio_std,
                                season_mask.sum()
                            )
                            sampled_ratios = np.clip(sampled_ratios, 1.0, 3.0)

                            df.loc[season_mask, 'FF_X'] = (
                                df.loc[season_mask, 'FF_AVG'] * sampled_ratios
                            )

                    logger.info(f"  Calculated {missing_ffx.sum()} FF_X values from FF_AVG ratio")

                # Cleanup temporary column
                df.drop('wind_ratio', axis=1, inplace=True, errors='ignore')

        # For remaining missing values (where FF_AVG is also missing)
        if df['FF_X'].isna().any():
            remaining = df['FF_X'].isna().sum()
            df['FF_X'] = df['FF_X'].interpolate(method='time', limit_direction='both')
            logger.info(f"  Interpolated {remaining} remaining FF_X values")

        # Ensure FF_X >= FF_AVG constraint
        if 'FF_AVG' in df.columns:
            violation_mask = df['FF_X'] < df['FF_AVG']
            if violation_mask.sum() > 0:
                logger.info(f"  Fixed {violation_mask.sum()} violations of FF_X >= FF_AVG")
                df.loc[violation_mask, 'FF_X'] = df.loc[violation_mask, 'FF_AVG'] * 1.2

        # Apply physical limits
        df['FF_X'] = df['FF_X'].clip(0, 50)

        # Track statistics
        self.preprocessing_stats['values_imputed']['FF_X'] = int(missing_count)
        self.preprocessing_stats['imputation_methods']['FF_X'] = 'seasonal_ratio_method'

        logger.info("  ✓ Maximum wind speed imputation completed")
        return df

    def _impute_sunshine_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute sunshine duration considering cloud cover (inferred from RH and RR)
        """
        logger.info("Step 3d: Imputing Sunshine Duration (SS)...")

        if 'SS' not in df.columns:
            logger.warning("  SS column not found")
            return df

        missing_count = df['SS'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing sunshine duration values detected")
            return df

        logger.info(f"  Found {missing_count} missing SS values")

        # Calculate maximum daylight hours based on latitude and day of year
        latitude = 5.5  # Aceh, Indonesia

        def max_daylight_hours(day_of_year, latitude):
            """Calculate max daylight using Cooper equation"""
            lat_rad = np.radians(latitude)
            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
            declination_rad = np.radians(declination)

            cos_hour_angle = -np.tan(lat_rad) * np.tan(declination_rad)
            cos_hour_angle = np.clip(cos_hour_angle, -1, 1)

            hour_angle = np.arccos(cos_hour_angle)
            day_length = 2 * hour_angle * 24 / (2 * np.pi)

            return day_length

        # Calculate day of year and max daylight
        df['day_of_year'] = df.index.dayofyear
        df['max_daylight'] = df['day_of_year'].apply(
            lambda d: max_daylight_hours(d, latitude)
        )

        # Estimate cloud cover from RH and rainfall
        if 'RH_AVG' in df.columns and 'RR' in df.columns:
            df['cloud_indicator'] = (
                (df['RH_AVG'] - 50) / 50 * 0.5 +  # High humidity → more clouds
                (df['RR'] > 0).astype(float) * 0.5   # Rain → definitely cloudy
            )
            df['cloud_indicator'] = df['cloud_indicator'].clip(0, 1)

            # Build predictive model if enough data
            predictors = ['max_daylight', 'cloud_indicator', 'RH_AVG', 'TAVG']
            available_predictors = [p for p in predictors if p in df.columns]

            train_mask = df['SS'].notna()
            for pred in available_predictors:
                train_mask = train_mask & df[pred].notna()

            if train_mask.sum() > 50:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import r2_score

                X_train = df.loc[train_mask, available_predictors]
                y_train = df.loc[train_mask, 'SS']

                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                logger.info("  Training sunshine duration model...")
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_train)
                r2 = r2_score(y_train, y_pred)
                logger.info(f"  Model R²: {r2:.4f}")

                # Predict missing values
                missing_mask = df['SS'].isna()
                X_missing = df.loc[missing_mask, available_predictors].copy()

                for col in X_missing.columns:
                    if X_missing[col].isna().any():
                        X_missing[col] = X_missing[col].fillna(df[col].median())

                predictions = model.predict(X_missing)

                # Apply constraint: SS <= max_daylight
                predictions = np.minimum(
                    predictions,
                    df.loc[missing_mask, 'max_daylight'].values
                )
                predictions = np.clip(predictions, 0, 14)

                df.loc[missing_mask, 'SS'] = predictions

                self.preprocessing_stats['imputation_methods']['SS'] = 'random_forest_cloud_aware'
                self.preprocessing_stats['model_performance']['sunshine_model'] = {
                    'r2_score': float(r2),
                    'features_used': available_predictors
                }

                logger.info(f"  Predicted {missing_mask.sum()} SS values using ML model")
            else:
                # Fallback to interpolation
                df['SS'] = df['SS'].interpolate(method='time', limit_direction='both')
                self.preprocessing_stats['imputation_methods']['SS'] = 'interpolation'

            # Cleanup temporary columns
            df.drop(['cloud_indicator'], axis=1, inplace=True, errors='ignore')
        else:
            # Simple interpolation if RH/RR not available
            df['SS'] = df['SS'].interpolate(method='time', limit_direction='both')
            self.preprocessing_stats['imputation_methods']['SS'] = 'interpolation'

        # Post-processing: Rainy days should have reduced sunshine
        if 'RR' in df.columns:
            rainy_days = df['RR'] > 5
            high_sunshine = df['SS'] > df['max_daylight'] * 0.7
            unrealistic_mask = rainy_days & high_sunshine

            if unrealistic_mask.sum() > 0:
                logger.info(f"  Adjusting {unrealistic_mask.sum()} rainy days with high sunshine")
                df.loc[unrealistic_mask, 'SS'] = (
                    df.loc[unrealistic_mask, 'max_daylight'] *
                    np.random.uniform(0.1, 0.4, unrealistic_mask.sum())
                )

        # Final constraints
        df['SS'] = df['SS'].clip(0, df['max_daylight'])

        # Cleanup
        df.drop(['day_of_year', 'max_daylight'], axis=1, inplace=True, errors='ignore')

        # Track statistics
        self.preprocessing_stats['values_imputed']['SS'] = int(missing_count)

        logger.info("  ✓ Sunshine duration imputation completed")
        return df

    def _impute_wind_direction_degrees(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute wind direction using circular statistics
        """
        logger.info("Step 3e: Imputing Wind Direction Degrees (DDD_X)...")

        if 'DDD_X' not in df.columns:
            logger.warning("  DDD_X column not found")
            return df

        missing_count = df['DDD_X'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing wind direction values detected")
            return df

        logger.info(f"  Found {missing_count} missing DDD_X values")

        def circular_mean(directions):
            """Calculate circular mean for wind directions"""
            if len(directions) == 0 or directions.isna().all():
                return np.nan

            directions_rad = np.radians(directions.astype(float))
            x_coords = np.cos(directions_rad)
            y_coords = np.sin(directions_rad)

            x_mean = np.nanmean(x_coords)
            y_mean = np.nanmean(y_coords)

            mean_direction = np.degrees(np.arctan2(y_mean, x_mean))
            return (mean_direction + 360) % 360

        def circular_consistency(directions):
            """Calculate consistency measure (0-1) for circular data"""
            if len(directions) == 0 or directions.isna().all():
                return 0

            directions_rad = np.radians(directions.astype(float))
            x_coords = np.cos(directions_rad)
            y_coords = np.sin(directions_rad)

            x_mean = np.nanmean(x_coords)
            y_mean = np.nanmean(y_coords)

            r = np.sqrt(x_mean**2 + y_mean**2)
            return r

        # Calculate monthly mean directions and consistency
        monthly_wind_data = {}
        for month in range(1, 13):
            month_data = df[df['month'] == month]['DDD_X'].dropna()
            if not month_data.empty:
                direction = circular_mean(month_data)
                consistency = circular_consistency(month_data)
                monthly_wind_data[month] = {
                    'direction': direction,
                    'consistency': consistency
                }

        logger.info("  Monthly wind direction patterns:")
        for month, data in monthly_wind_data.items():
            logger.info(f"    Month {month}: {data['direction']:.1f}° (consistency: {data['consistency']:.2f})")

        # Impute using monthly means (only if consistency is good)
        for idx in df.index[df['DDD_X'].isna()]:
            month = idx.month if hasattr(idx, 'month') else df.loc[idx, 'Month']

            if month in monthly_wind_data and monthly_wind_data[month]['consistency'] > 0.5:
                df.loc[idx, 'DDD_X'] = monthly_wind_data[month]['direction']

        # For remaining missing values, use nearest neighbor (within 7 days)
        if df['DDD_X'].isna().any():
            remaining = df['DDD_X'].isna().sum()

            for idx in df.index[df['DDD_X'].isna()]:
                # Find closest date with valid wind direction
                closest_idx = None
                min_diff = pd.Timedelta(days=7)

                for other_idx in df.index[~df['DDD_X'].isna()]:
                    diff = abs(other_idx - idx)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = other_idx

                if closest_idx is not None:
                    df.loc[idx, 'DDD_X'] = df.loc[closest_idx, 'DDD_X']

            logger.info(f"  Used nearest neighbor for {remaining} values")

        # Final fallback: overall circular mean
        if df['DDD_X'].isna().any():
            overall_mean = circular_mean(df['DDD_X'].dropna())
            df['DDD_X'] = df['DDD_X'].fillna(overall_mean)

        # Ensure values are in [0, 360)
        df['DDD_X'] = df['DDD_X'] % 360

        # Track statistics
        self.preprocessing_stats['values_imputed']['DDD_X'] = int(missing_count)
        self.preprocessing_stats['imputation_methods']['DDD_X'] = 'circular_statistics'

        logger.info("  ✓ Wind direction imputation completed")
        return df

    def _impute_wind_direction_cardinal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute cardinal wind direction from DDD_X or using mode
        """
        logger.info("Step 3f: Imputing Cardinal Wind Direction (DDD_CAR)...")

        if 'DDD_CAR' not in df.columns:
            logger.warning("  DDD_CAR column not found")
            return df

        # Convert to string type
        df['DDD_CAR'] = df['DDD_CAR'].replace({None: np.nan, 'None': np.nan})
        df['DDD_CAR'] = df['DDD_CAR'].astype(str)
        df.loc[df['DDD_CAR'].isin(['nan', 'None', 'null']), 'DDD_CAR'] = np.nan

        missing_count = df['DDD_CAR'].isna().sum()

        if missing_count == 0:
            logger.info("  No missing cardinal direction values detected")
            return df

        logger.info(f"  Found {missing_count} missing DDD_CAR values")

        def degrees_to_cardinal(degrees):
            """Convert degrees to cardinal direction"""
            if np.isnan(degrees):
                return np.nan

            dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

            idx = round(degrees / 22.5) % 16
            return dirs[idx]

        # Convert from DDD_X where available
        if 'DDD_X' in df.columns:
            missing_car = df['DDD_CAR'].isna() & df['DDD_X'].notna()
            if missing_car.sum() > 0:
                df.loc[missing_car, 'DDD_CAR'] = df.loc[missing_car, 'DDD_X'].apply(degrees_to_cardinal)
                logger.info(f"  Converted {missing_car.sum()} directions from DDD_X")

        # For remaining, use monthly mode
        for month in range(1, 13):
            month_data = df[df['month'] == month]['DDD_CAR'].dropna()
            if not month_data.empty:
                month_mode = month_data.mode()[0]
                month_mask = (df['month'] == month) & df['DDD_CAR'].isna()
                if month_mask.sum() > 0:
                    df.loc[month_mask, 'DDD_CAR'] = month_mode

        # Final fallback: overall mode
        if df['DDD_CAR'].isna().any():
            overall_mode = df['DDD_CAR'].value_counts().idxmax() if not df['DDD_CAR'].dropna().empty else 'N'
            df['DDD_CAR'] = df['DDD_CAR'].fillna(overall_mode)

        # Track statistics
        self.preprocessing_stats['values_imputed']['DDD_CAR'] = int(missing_count)
        self.preprocessing_stats['imputation_methods']['DDD_CAR'] = 'conversion_and_mode'

        logger.info("  ✓ Cardinal direction imputation completed")
        return df

    def preprocess(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess BMKG dataset with comprehensive statistics tracking
        """
        try:
            start_time = datetime.now()

            # Default options
            default_options = {
                "location": "Aceh",
                "latitude": 5.5,
                "save_plots": False,
                "validate_results": True
            }

            if options is None:
                options = {}
            self.options = {**default_options, **options}

            logger.info("=" * 70)
            logger.info("BMKG DATA PREPROCESSING PIPELINE")
            logger.info("=" * 70)

            # Step 1: Validate dataset
            logger.info("\n[1/5] Validating dataset...")
            validation_result = self.validator.validate_dataset(self.db, self.collection_name)

            if not validation_result.get('valid', False):
                raise BmkgPreprocessingError(
                    f"Validation failed: {validation_result.get('errors', ['Unknown error'])}"
                )

            logger.info(f"✓ Validation passed - {validation_result['total_records']} records")

            # Step 2: Load data
            logger.info("\n[2/5] Loading data from MongoDB...")
            df = self.loader.load_data(self.db, self.collection_name)
            original_record_count = len(df)
            logger.info(f"✓ Loaded {original_record_count} records")

            # Step 3: Apply preprocessing
            logger.info("\n[3/5] Applying preprocessing...")
            processed_df = self._apply_preprocessing(df)
            logger.info("✓ Preprocessing completed")

            # Step 4: Save processed data
            logger.info("\n[4/5] Saving preprocessed data...")
            save_result = self.saver.save_preprocessed_data(
                self.db,
                processed_df,
                self.collection_name
            )
            logger.info(f"✓ Saved to collection: {save_result['preprocessedCollections'][0]}")

            # Step 5: Generate summary
            logger.info("\n[5/5] Generating summary...")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Calculate data quality score
            total_cells = original_record_count * 10  # 10 main variables
            total_imputed = sum(self.preprocessing_stats['values_imputed'].values())
            data_quality_score = 1 - (total_imputed / total_cells) if total_cells > 0 else 1.0

            result = {
                "status": "success",
                "message": "BMKG dataset preprocessed successfully",
                "collection": self.collection_name,
                "cleanedCollection": save_result.get("preprocessedCollections", [])[0],
                "recordCount": len(processed_df),
                "originalRecordCount": original_record_count,
                "processingTime": round(processing_time, 2),

                "preprocessing_summary": {
                    "outliers_removed": self.preprocessing_stats['outliers_removed'],
                    "values_imputed": self.preprocessing_stats['values_imputed'],
                    "imputation_methods": self.preprocessing_stats['imputation_methods'],
                    "total_values_imputed": total_imputed,
                    "data_quality_score": round(data_quality_score, 4)
                },

                "model_performance": self.preprocessing_stats['model_performance'],

                "validation_info": {
                    "total_records": validation_result['total_records'],
                    "temporal_coverage": validation_result.get('temporal_info', {}),
                    "field_statistics": validation_result.get('field_statistics', {}),
                    "warnings": validation_result.get('warnings', [])
                },

                "sample_data": processed_df.head(10).to_dict('records'),
                "metadata": save_result.get("metadata", {})
            }

            logger.info("✓ Summary generated")
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Data quality score: {data_quality_score:.4f}")

            return result

        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise BmkgPreprocessingError(error_msg)