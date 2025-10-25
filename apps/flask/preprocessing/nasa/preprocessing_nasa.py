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
            if "dataset-meta" in db.list_collection_names():
                meta_collection = "dataset-meta"
            elif "DatasetMeta" in db.list_collection_names():
                meta_collection = "DatasetMeta"
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
            logger.info(f"Updated metadata for collection '{original_collection_name}")
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
                "smoothing_method": "moving_average",
                "window_size": 3,
                "drop_outliers": True,
                "fill_missing": True,
                "columns_to_process": [
                    'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 
                    'WS10M', 'WS10M_MAX', 'WD10M'
                ]
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
                "preprocessedData": processed_df.head(5).to_dict('records'),  # Sample for preview
                "recordCount": len(processed_df),
                "preprocessedCollections": save_result.get("preprocessedCollections", []),
                "cleanedCollection": save_result.get("preprocessedCollections", [])[0] if save_result.get("preprocessedCollections") else None,
                "warnings": []  # Will be populated if any warnings occur during preprocessing
            }
        except Exception as e:
            error_msg = f"Error preprocessing NASA POWER data: {str(e)}"
            logger.error(error_msg)
            raise NasaPreprocessingError(error_msg)
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to the data
        
        Note: This method will be implemented in the future with actual preprocessing logic.
        For now, it just returns the original dataframe.
        """
        logger.info("Preprocessing NASA POWER data with smoothing methods (placeholder)")
        
        # TODO: Implement actual preprocessing steps here
        # - This will include smoothing methods for each parameter
        # - Currently just returning the original dataframe

        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Here is where we would implement smoothing and other preprocessing
        # For now, just return the original data
        
        logger.info(f"Preprocessing completed - processed {len(processed_df)} records")
        return processed_df