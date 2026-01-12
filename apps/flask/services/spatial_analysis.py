import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pymongo import MongoClient
from pymongo.database import Database
from bson import ObjectId
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NasaDatasetInfo:
    """Information about a NASA POWER dataset"""
    collection_name: str
    name: str
    latitude: float
    longitude: float
    date_range: Tuple[datetime, datetime]
    parameters: List[str]
    total_records: int
    
class NasaSpatialConnector:
    """Connects to MongoDB and retrieves NASA POWER datasets for spatial analysis"""
    
    def __init__(self, db: Database):
        self.db = db
        self.logger = logging.getLogger(__name__)
        
    def get_nasa_datasets(self) -> List[NasaDatasetInfo]:
        """
        Get all available NASA POWER datasets from metadata collection
        Only returns datasets with source containing 'nasa' and isAPI=True
        """
        try:
            self.logger.info("🔍 Searching for NASA POWER datasets...")
            
            # Find metadata collection (handle both naming conventions)
            meta_collection = None
            if "dataset_meta" in self.db.list_collection_names():
                meta_collection = self.db["dataset_meta"]
            elif "DatasetMeta" in self.db.list_collection_names():
                meta_collection = self.db["DatasetMeta"]
            else:
                raise Exception("No dataset metadata collection found")
            
            # Query for NASA POWER datasets
            query = {
                "$and": [
                    {"source": {"$regex": "nasa", "$options": "i"}},  # Case-insensitive NASA source
                    {"isAPI": True},  # Only API datasets (NASA POWER)
                    {"status": {"$in": ["raw", "latest", "preprocessed", "validated"]}},  # Exclude archived
                    {"deletedAt": None}  # Not deleted
                ]
            }
            
            nasa_datasets = list(meta_collection.find(query))
            
            if not nasa_datasets:
                self.logger.warning("⚠️ No NASA POWER datasets found")
                return []
            
            self.logger.info(f"📊 Found {len(nasa_datasets)} NASA POWER datasets")
            
            # Process each dataset to extract spatial info
            dataset_infos = []
            for dataset in nasa_datasets:
                try:
                    info = self._extract_dataset_info(dataset)
                    if info:
                        dataset_infos.append(info)
                        self.logger.debug(f"✅ Processed dataset: {info.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to process dataset {dataset.get('name', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"🎯 Successfully processed {len(dataset_infos)} NASA POWER datasets")
            return dataset_infos
            
        except Exception as e:
            self.logger.error(f"❌ Error getting NASA datasets: {str(e)}")
            raise Exception(f"Failed to retrieve NASA datasets: {str(e)}")
        
    def _extract_dataset_info(self, dataset: Dict) -> Optional[NasaDatasetInfo]:
        """Extract spatial information from dataset metadata"""
        try:
            # Get basic info
            collection_name = dataset.get('collectionName')
            name = dataset.get('name', '')
            
            if not collection_name:
                raise ValueError("Missing collection name")
            
            # Verify collection exists
            if collection_name not in self.db.list_collection_names():
                raise ValueError(f"Collection {collection_name} does not exist")
            
            # Get coordinates from apiConfig
            api_config = dataset.get('apiConfig', {})
            params = api_config.get('params', {})
            
            latitude = params.get('latitude')
            longitude = params.get('longitude')
            
            if latitude is None or longitude is None:
                raise ValueError("Missing latitude or longitude in apiConfig")
            
            # Convert to float
            latitude = float(latitude)
            longitude = float(longitude)
            
            # Validate coordinates (should be in Aceh region approximately)
            if not (2.0 <= latitude <= 6.0) or not (95.0 <= longitude <= 98.0):
                self.logger.warning(f"⚠️ Coordinates outside Aceh region: lat={latitude}, lng={longitude}")
            
            # Get parameters
            parameters = params.get('parameters', [])
            if not parameters:
                raise ValueError("No climate parameters found")
            
            # Get date range from actual data
            collection = self.db[collection_name]
            
            # Find min and max dates
            date_pipeline = [
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$Date"},
                    "max_date": {"$max": "$Date"},
                    "total_records": {"$sum": 1}
                }}
            ]
            
            date_result = list(collection.aggregate(date_pipeline))
            
            if not date_result:
                raise ValueError("No data found in collection")
            
            date_info = date_result[0]
            min_date = date_info.get('min_date')
            max_date = date_info.get('max_date')
            total_records = date_info.get('total_records', 0)
            
            if not min_date or not max_date:
                raise ValueError("Invalid date range in data")
            
            # Ensure dates are datetime objects
            if not isinstance(min_date, datetime):
                min_date = datetime.fromisoformat(str(min_date))
            if not isinstance(max_date, datetime):
                max_date = datetime.fromisoformat(str(max_date))
            
            return NasaDatasetInfo(
                collection_name=collection_name,
                name=name,
                latitude=latitude,
                longitude=longitude,
                date_range=(min_date, max_date),
                parameters=parameters,
                total_records=total_records
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting dataset info: {str(e)}")
            return None
        
    def get_datasets_summary(self) -> Dict[str, Any]:
        """Get summary of all available NASA POWER datasets"""
        try:
            datasets = self.get_nasa_datasets()
            
            if not datasets:
                return {
                    "total_datasets": 0,
                    "datasets": [],
                    "coverage_area": None,
                    "date_range": None,
                    "available_parameters": []
                }
            
            # Calculate coverage area
            latitudes = [d.latitude for d in datasets]
            longitudes = [d.longitude for d in datasets]
            
            coverage_area = {
                "bounds": {
                    "north": max(latitudes),
                    "south": min(latitudes),
                    "east": max(longitudes),
                    "west": min(longitudes)
                },
                "center": {
                    "latitude": sum(latitudes) / len(latitudes),
                    "longitude": sum(longitudes) / len(longitudes)
                }
            }
            
            # Calculate overall date range
            all_start_dates = [d.date_range[0] for d in datasets]
            all_end_dates = [d.date_range[1] for d in datasets]
            
            date_range = {
                "earliest": min(all_start_dates),
                "latest": max(all_end_dates)
            }
            
            # Get all unique parameters
            all_parameters = set()
            for dataset in datasets:
                all_parameters.update(dataset.parameters)
            
            # Prepare dataset list
            dataset_list = []
            for dataset in datasets:
                dataset_list.append({
                    "collection_name": dataset.collection_name,
                    "name": dataset.name,
                    "coordinates": {
                        "latitude": dataset.latitude,
                        "longitude": dataset.longitude
                    },
                    "date_range": {
                        "start": dataset.date_range[0],
                        "end": dataset.date_range[1]
                    },
                    "parameters": dataset.parameters,
                    "total_records": dataset.total_records
                })
            
            return {
                "total_datasets": len(datasets),
                "datasets": dataset_list,
                "coverage_area": coverage_area,
                "date_range": date_range,
                "available_parameters": sorted(list(all_parameters)),
                "total_records": sum(d.total_records for d in datasets)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting datasets summary: {str(e)}")
            raise Exception(f"Failed to get datasets summary: {str(e)}")
        
        
# Helper function for flask routes
def create_spatial_connector(db: Database) -> NasaSpatialConnector:
    """Create a spatial connector instance"""
    return NasaSpatialConnector(db)