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
                logger.debug(f"Found collection {collection_name} for {nasa_match}")
                return collection_name
            
            logger.warning(f"No collection found for NASA location: {nasa_match}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting collection name for {nasa_match}: {str(e)}")
            return None
    
    def load_climate_data(
        self, 
        collection_name: str,
        start_year: int,
        end_year: int
    ) -> Optional [pd.DataFrame]:
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
            
            logger.info(f"Loading climate data from {collection_name}: {start_year}-{end_year}")
            
            # Load data into DataFrame
            cursor = collection.find(query)
            data = list(cursor)
            
            if not data:
                logger.warning(f"No data found in {collection_name} for period {start_year}-{end_year}")
                return None
            
            df = pd.DataFrame(data)
            
            # Convert Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            
            logger.info(f"Loaded {len(df)} records from {collection_name}")
            
            return df            
        except Exception as e:
            logger.error(f"Error loading climate data from {collection_name}: {str(e)}")
            return None            
    
    def apply_seasonal_filter(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Filter data by season (wet/dry/all)"""
        try:
            if season == 'all':
                return df
                
            # Ensure Month column exists
            if 'Month' not in df.columns:
                if 'Date' in df.columns:
                    df['Month'] = pd.to_datetime(df['Date']).dt.month
                else:
                    logger.warning("No Date column available for seasonal filtering")
                    return df
            
            if season == 'wet':
                # Wet season: October - March (months 10, 11, 12, 1, 2, 3)
                filtered_df = df[df['Month'].isin([10, 11, 12, 1, 2, 3])]
            elif season == 'dry': 
                # Dry season: April - September (months 4, 5, 6, 7, 8, 9)
                filtered_df = df[df['Month'].isin([4, 5, 6, 7, 8, 9])]
            else:
                logger.warning(f"Unknown season filter: {season}")
                return df
            
            logger.info(f"Applied {season} season filter: {len(filtered_df)}/{len(df)} records")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying seasonal filter: {str(e)}")
            return df
    
    def aggregate_climate_data(self, df: pd.DataFrame, method: str = 'mean') -> Dict[str, float]:
        """Aggregate climate parameters using specified method"""
        try:
            # Key NASA POWER parameters for rice suitability
            climate_params = {
                'T2M': 'temperature',          # Temperature at 2m (°C)
                'T2M_MAX': 'max_temperature',   # Maximum temperature
                'T2M_MIN': 'min_temperature',   # Minimum temperature  
                'PRECTOTCORR': 'precipitation', # Precipitation (mm/day)
                'RH2M': 'humidity',             # Relative humidity (%)
                'ALLSKY_SFC_SW_DWN': 'solar',   # Solar radiation
                'WS10M': 'wind_speed'           # Wind speed (m/s)
            }
            
            aggregated = {}
            
            for param, name in climate_params.items():
                if param in df.columns:
                    values = df[param].dropna()
                    
                    if len(values) == 0:
                        aggregated[param] = 0.0
                        continue
                    
                    if method == 'mean':
                        aggregated[param] = float(values.mean())
                    elif method == 'median':
                        aggregated[param] = float(values.median())
                    elif method == 'percentile_90':
                        aggregated[param] = float(values.quantile(0.9))
                    elif method == 'percentile_10':
                        aggregated[param] = float(values.quantile(0.1))
                    else:
                        # Default to mean
                        aggregated[param] = float(values.mean())
                else:
                    aggregated[param] = 0.0
            
            # Calculate derived metrics
            if 'T2M_MAX' in aggregated and 'T2M_MIN' in aggregated:
                aggregated['temperature_range'] = aggregated['T2M_MAX'] - aggregated['T2M_MIN']
            
            # Convert daily precipitation to annual (mm/day * 365)
            if 'PRECTOTCORR' in aggregated:
                aggregated['annual_precipitation'] = aggregated['PRECTOTCORR'] * 365
            
            logger.info(f"Aggregated {len(aggregated)} climate parameters using {method}")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating climate data: {str(e)}")
            return {}
    
    def get_available_parameters(self, collection_name: str) -> List[str]:
        """Get list of available climate parameters in collection"""
        try:
            if collection_name not in self.db.list_collection_names():
                return []
            
            collection = self.db[collection_name]
            
            # Get one document to check available fields
            sample_doc = collection.find_one()
            
            if not sample_doc:
                return []
            
            # Filter climate-related parameters
            climate_params = []
            exclude_fields = ['_id', 'Date', 'Year', 'Month']
            
            for field in sample_doc.keys():
                if field not in exclude_fields:
                    climate_params.append(field)
            
            logger.info(f"Found {len(climate_params)} climate parameters in {collection_name}")
            
            return sorted(climate_params)
            
        except Exception as e:
            logger.error(f"Error getting available parameters: {str(e)}")
            return []
    
    def get_climate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed climate statistics"""
        try:
            stats = {
                'record_count': len(df),
                'date_range': {
                    'start': df['Date'].min().isoformat() if 'Date' in df.columns else None,
                    'end': df['Date'].max().isoformat() if 'Date' in df.columns else None
                },
                'parameters': {}
            }
            
            # Calculate statistics for each numerical column
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['_id']:  # Skip non-climate columns
                    continue
                    
                values = df[col].dropna()
                
                if len(values) > 0:
                    stats['parameters'][col] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': int(len(values))
                    }
            
            logger.info(f"Calculated statistics for {len(stats['parameters'])} parameters")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate climate data quality"""
        try:
            quality_report = {
                'total_records': len(df),
                'missing_data': {},
                'outliers': {},
                'data_completeness': 0.0,
                'quality_score': 0.0
            }
            
            # Check for missing data
            for col in df.columns:
                if col not in ['_id', 'Date']:
                    missing_count = df[col].isna().sum()
                    missing_percentage = (missing_count / len(df)) * 100
                    quality_report['missing_data'][col] = {
                        'count': int(missing_count),
                        'percentage': float(missing_percentage)
                    }
            
            # Calculate overall data completeness
            total_cells = len(df) * (len(df.columns) - 1)  # Exclude _id
            missing_cells = sum(df[col].isna().sum() for col in df.columns if col != '_id')
            quality_report['data_completeness'] = ((total_cells - missing_cells) / total_cells) * 100
            
            # Simple quality score based on completeness
            quality_report['quality_score'] = quality_report['data_completeness']
            
            logger.info(f"Data quality validation completed: {quality_report['quality_score']:.1f}% quality score")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {}