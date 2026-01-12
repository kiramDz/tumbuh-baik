import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo.database import Database

logger = logging.getLogger(__name__)

class ClimateDataService:
    """Service for climate data loading and processing operations"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def get_collection_name_for_nasa_location(self, nasa_match: str) -> Optional[str]:
        """Get MongoDB collection name for NASA location"""
        try:
            # Query metadata to find collection for this NASA location
            meta_collection = None
            if "dataset_meta" in self.db.list_collection_names():
                meta_collection = self.db["dataset_meta"]
            elif "DatasetMeta" in self.db.list_collection_names():
                meta_collection = self.db["DatasetMeta"]
                
            if meta_collection is None:
                logger.warning("No dataset metadata collection found")
                return None
            
            # Find dataset with matching name
            query = {
                "$and": [
                    {"source": {"$regex": "nasa", "$options": "i"}},
                    {"name": {"$regex": nasa_match, "$options": "i"}},
                    {"isAPI": True},
                    {"deletedAt": None}
                ]
            }
            
            dataset = meta_collection.find_one(query)
            
            if dataset:
                collection_name = dataset.get('collectionName')
                logger.info(f"Found collection {collection_name} for {nasa_match}")
                return collection_name
            
            logger.warning(f"No collection found for {nasa_match}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting collection name: {str(e)}")
            return None
    
    def load_climate_data(
        self, 
        collection_name: str,
        start_year: int,
        end_year: int
    ) -> Optional[pd.DataFrame]:
        """Load NASA POWER climate data for specified period"""
        try:
            # Check if collection exists
            if collection_name not in self.db.list_collection_names():
                logger.warning(f"Collection {collection_name} not found")
                return None
            
            collection = self.db[collection_name]
            
            # Build date filter
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            
            # Query data
            query = {
                "Date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            logger.info(f"Loading climate data from {collection_name}")
            
            # Load data into DataFrame
            cursor = collection.find(query)
            data = list(cursor)
            
            if not data:
                logger.warning(f"No data found in {collection_name}")
                return None
            
            df = pd.DataFrame(data)
            
            # Convert Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            logger.info(f"Loaded {len(df)} records")
            
            return df            
        except Exception as e:
            logger.error(f"Error loading climate data: {str(e)}")
            return None
    
    def aggregate_climate_data(self, df: pd.DataFrame, aggregation: str = 'mean') -> Dict[str, float]:
        """Aggregate key climate parameters using specified method"""
        try:
            # Key parameters needed for rice analysis
            params = ['T2M', 'PRECTOTCORR', 'RH2M']  # Added RH2M for humidity
            
            aggregated = {}
            
            for param in params:
                if param in df.columns:
                    values = df[param].dropna()
                    if len(values) > 0:
                        if aggregation == 'mean':
                            aggregated[param] = float(values.mean())
                        elif aggregation == 'median':
                            aggregated[param] = float(values.median())
                        elif aggregation == 'max':
                            aggregated[param] = float(values.max())
                        elif aggregation == 'min':
                            aggregated[param] = float(values.min())
                        else:
                            aggregated[param] = float(values.mean())  # Default to mean
                    else:
                        aggregated[param] = 0.0
                else:
                    logger.warning(f"Parameter {param} not found in data")
                    # Provide fallback values for missing parameters
                    if param == 'RH2M':
                        aggregated[param] = 75.0  # Default humidity for Aceh
                    else:
                        aggregated[param] = 0.0
            
            # Convert daily precipitation to annual
            if 'PRECTOTCORR' in aggregated:
                aggregated['annual_precipitation'] = aggregated['PRECTOTCORR'] * 365
            
            logger.info(f"Aggregated climate data using {aggregation}: T2M={aggregated.get('T2M', 0):.1f}°C, "
                    f"PRECTOTCORR={aggregated.get('PRECTOTCORR', 0):.1f}mm/day, "
                    f"RH2M={aggregated.get('RH2M', 0):.1f}%")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating climate data: {str(e)}")
            return {
                'T2M': 26.0,           # Default temperature for Aceh
                'PRECTOTCORR': 8.0,    # Default precipitation  
                'RH2M': 75.0,          # Default humidity
                'annual_precipitation': 2920.0
            }