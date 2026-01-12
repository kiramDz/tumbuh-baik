import geopandas as gpd
import pandas as pd
import logging
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)

class GeojsonLoader:
    """Service for loading and managing filtered kecamatan GeoJSON"""
    
    def __init__(self):
        self.geojson_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/aceh_nasa_kecamatan.geojson"
        self.summary_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/nasa_kecamatan_summary.json"

    def load_kecamatan_geojson(self) -> Optional[gpd.GeoDataFrame]:
        """Load filtered kecamatan GeoJSON with NASA matches"""
        try:
            logger.info(f"Loading kecamatan GeoJSON from: {self.geojson_path}")
            kecamatan_gdf = gpd.read_file(self.geojson_path)
            logger.info(f"Loaded {len(kecamatan_gdf)} kecamatan")
            return kecamatan_gdf
        except Exception as e:
            logger.error(f"Error loading GeoJSON: {str(e)}")
            return None
    
    def load_summary(self) -> Optional[Dict]:
        """Load kecamatan summary information"""
        try:
            with open(self.summary_path, 'r') as f:
                summary = json.load(f)
            logger.info("Loaded kecamatan summary")
            return summary
        except Exception as e:
            logger.error(f"Error loading summary: {str(e)}")
            return None
        
    def get_nasa_location_mapping(self) -> Dict[str, Dict]:
        """Get simple mapping of NASA match names to coordinates"""
        try:
            kecamatan_gdf = self.load_kecamatan_geojson()
            if kecamatan_gdf is None:
                return {}
            
            mapping = {}
            for _, row in kecamatan_gdf.iterrows():
                nasa_match = row['nasa_match']
                mapping[nasa_match] = {
                    'kecamatan': row['NAME_3'],
                    'kabupaten': row['NAME_2'],
                    'lat': row['nasa_lat'],
                    'lng': row['nasa_lng']
                }
            
            logger.info(f"Created mapping for {len(mapping)} locations")
            return mapping
            
        except Exception as e:
            logger.error(f"Error creating location mapping: {str(e)}") 
            return {}
        
    def get_districts_info(self) -> List[Dict]:
        """Get district information with simplified coordinates"""
        try:
            kecamatan_gdf = self.load_kecamatan_geojson()
            if kecamatan_gdf is None:
                return []
            
            districts = []
            for _, row in kecamatan_gdf.iterrows():
                districts.append({
                    "name": row['NAME_3'],
                    "kabupaten": row['NAME_2'],
                    "nasa_match": row['nasa_match'],
                    "coordinates": {
                        "lat": row['nasa_lat'],
                        "lng": row['nasa_lng']
                    }
                })
            
            logger.info(f"Retrieved {len(districts)} districts")
            return districts
            
        except Exception as e:
            logger.error(f"Error getting districts info: {str(e)}")
            return []
    
    def get_nasa_matches(self) -> List[str]:
        """Get list of all NASA location matches"""
        try:
            gdf = self.load_kecamatan_geojson()
            if gdf is None:
                return []
            
            nasa_matches = gdf['nasa_match'].unique().tolist()
            logger.info(f"Found {len(nasa_matches)} unique NASA matches")
            return nasa_matches
            
        except Exception as e:
            logger.error(f"Error getting NASA matches: {str(e)}")
            return []