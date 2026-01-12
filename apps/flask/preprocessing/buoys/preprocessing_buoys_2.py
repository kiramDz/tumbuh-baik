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

class BuoysPreprocessingError(Exception):
    """Custom exception for Buoys preprocessing errors"""
    pass

class BuoysDataValidator:
    """Validates Buoys data before preprocessing"""
    
    def validate_dataset(self, db, collection_name: str) -> Dict[str, Any]:
        """
        Validates that the dataset contains required columns and is suitable for preprocessing
        Returns a dictionary with validation results
        """
        try:
            # Check if collection exists
            if collection_name not in db.list_collection_names():
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} does not exist"]
                }
            
            # Get sample document to check schema
            sample = db[collection_name].find_one()
            if not sample:
                return {
                    'valid': False,
                    'errors': [f"Collection {collection_name} is empty"]
                }
            
            # Define required fields for Buoys dataset based on your sample
            required_fields = [
                'SST',      # Sea Surface Temperature
                'Prec',     # Precipitation  
                'RH',       # Relative Humidity
                'WSPD',     # Wind Speed
                'SWRad'     # Solar Radiation
            ]
            
            # Check if all required fields exist
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing required fields: {', '.join(missing_fields)}"]
                }
            
            # Check if date field exists
            date_fields = ['Date', 'year', 'month', 'day']
            missing_date_fields = [field for field in date_fields if field not in sample]
            
            if missing_date_fields:
                return {
                    'valid': False,
                    'errors': [f"Missing date fields: {', '.join(missing_date_fields)}"]
                }
            
            # Get total record count for validation
            total_records = db[collection_name].count_documents({})
            
            # All validations passed
            return {
                'valid': True,
                'message': "Buoy dataset is valid for preprocessing!",
                'total_records': total_records
            }
            
        except Exception as e:
            logger.error(f"Error during dataset validation: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
            
class BuoysDataLoader:
    """Loads Buoys data from MongoDB into pandas DataFrame"""
    
    def load_data(self, db, collection_name: str) -> pd.DataFrame:
        """Load all data from MongoDB into pd.DataFrame"""
        try:
            # Load all records from MongoDB collection, no pagination
            cursor = db[collection_name].find({})
            df = pd.DataFrame(list(cursor))
            
            if len(df) == 0:
                raise BuoysPreprocessingError(f"No data found in collection '{collection_name}'")
            
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
            raise BuoysPreprocessingError(error_msg)

class BuoysDataSaver:
    """Saves preprocessed Buoys data back to new collection in MongoDB"""

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
            
            # Remove _id column - MongoDB will auto-generate new IDs
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
            raise BuoysPreprocessingError(error_msg)
    
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
            
            # Get cleaned collection columns and filter technical fields
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
                "collectionName": cleaned_collection_name,  # Point to cleaned collection
                "totalRecords": record_count,
                "columns": cleaned_columns,  # Filtered columns
                "fileType": "json",  # Updated from original format
                "lastUpdated": datetime.now(),
                "name": f"{original_meta.get('name', original_collection_name)} (Cleaned)"
            }
            
            # For buoy datasets (isAPI: false), do NOT include apiConfig
            # This is different from NASA preprocessing
            
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

class BuoysPreprocessor:
    """Main class for preprocessing Buoys datasets"""

    def __init__(self, db, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.validator = BuoysDataValidator()
        self.loader = BuoysDataLoader()
        self.saver = BuoysDataSaver()
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
        Preprocess Buoy dataset with comprehensive statistics tracking
        
        Args:
            options: Dictionary of preprocessing options (provided by user - no defaults)
            
        Returns:
            Dictionary with preprocessing results and processed dataframe
        """
        try:
            start_time = datetime.now()
            
            # Use provided options or empty dict (no defaults for manual uploads)
            if options is None:
                options = {}
            self.options = options
            
            logger.info("BUOY DATA PREPROCESSING PIPELINE")
            
            # Step 1: Validate dataset
            logger.info("\n[1/5] Validating dataset...")
            validation_result = self.validator.validate_dataset(self.db, self.collection_name)
            
            if not validation_result.get('valid', False):
                raise BuoysPreprocessingError(
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
            
            # Step 4: Save processed data
            logger.info("\n[4/5] Saving preprocessed data...")
            save_result = self.saver.save_preprocessed_data(
                self.db, 
                processed_df, 
                self.collection_name
            )
            
            # Step 5: Generate final report
            logger.info("\n[5/5] Generating final report...")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "status": "success",
                "message": "Buoy dataset preprocessed successfully",
                "collection": self.collection_name,
                "preprocessedData": processed_df.head(10).to_dict('records'),
                "recordCount": len(processed_df),
                "originalRecordCount": original_record_count,
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None,
                "preprocessing_report": self.preprocessing_report,
                "processing_time_seconds": round(processing_time, 2),
                "warnings": self.preprocessing_report.get("warnings", [])
            }
            
        except Exception as e:
            error_msg = f"Error preprocessing Buoy data: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise BuoysPreprocessingError(error_msg)
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the data with progress tracking"""
        logger.info("Starting Buoys data preprocessing...")

        # TODO: Implement actual preprocessing steps here
        # Based on your buoy data structure (SST, Prec, RH, WSPD, SWRad)
        # For now, just return the original dataframe
        
        processed_df = df.copy()
        
        # Here is where we would implement:
        # - Fill value replacement for buoy data
        # - Missing value imputation
        # - Outlier detection and treatment  
        # - Smoothing algorithms
        # - Quality validation
        # - Parameter-specific processing
        
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df