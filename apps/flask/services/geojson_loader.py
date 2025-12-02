import geopandas as gpd
import pandas as pd
import logging
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)

class GeojsonLoader:
    """Service for loading and managing filtered kecamatan GeoJSON"""
    
    def __init__(self):
        self.geojson_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/tumbuh-baik/apps/flask/data/geojson/aceh_nasa_kecamatan.geojson"
        self.summary_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/tumbuh-baik/apps/flask/data/geojson/nasa_kecamatan_summary.json"
        
    def load_kecamatan_geojson(self) -> Optional[gpd.GeoDataFrame]:
        """Load filtered kecamatan GeoJSON with NASA matches"""

        try:
            logger.info(f"Loading kecamatan GeoJSON from: {self.geojson_path}")
            kecamatan_gdf = gpd.read_file(self.geojson_path)
            logger.info(f"Successfully loaded {len(kecamatan_gdf)} kecamatan NASA matches.")
            return kecamatan_gdf
        except FileNotFoundError:
            logger.error(f"GeoJSON file not found: {self.geojson_path}")
            return None
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
        except FileNotFoundError:
            logger.warning(f"Summary file not found: {self.summary_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading summary: {str(e)}")
            return None
        
    def get_nasa_location_mapping(self) -> Dict[str, str]:
        """Get mapping of NASA match names to kecamatan"""
        
        try:
            kecamatan_gdf = self.load_kecamatan_geojson()
            if kecamatan_gdf is None:
                return {}
            
            mapping = {}
            for _, row in kecamatan_gdf.iterrows():
                nasa_match = row['nasa_match']
                kecamatan_name = row['NAME_3']
                kabupaten_name = row['NAME_2']
                
                mapping[nasa_match] = {
                    'kecamatan': kecamatan_name,
                    'kabupaten': kabupaten_name,
                    'gid': row['GID_3'] 
                }
            
            logger.info(f"Created NASA location mapping for {len(mapping)} locations")
            return mapping
            
        except Exception as e:
            logger.info(f"Error creating NASA location mapping: {str(e)}") 
            return {}
        
    def get_districts_info(self) -> List[Dict]:
        """Get detailed information about all districts"""
        try:
            kecamatan_gdf = self.load_kecamatan_geojson()
            
            if kecamatan_gdf is None:
                return []
            
            districts = []
            for _, row in kecamatan_gdf.iterrows():
                districts.append({
                    "id": row['GID_3'],
                    "name": row['NAME_3'],
                    "kabupaten": row['NAME_2'],
                    "nasa_match": row['nasa_match'],
                    "coordinates": {
                        "nasa_point": {
                            "latitude": row['nasa_lat'],
                            "longitude": row['nasa_lng']
                        },
                        "centroid": {
                            "latitude": row['centroid_lat'],
                            "longitude": row['centroid_lng']
                        }
                    }
                })
            
            logger.info(f"Retrieved {len(districts)} districts info")
            
            return districts
            
        except Exception as e:
            logger.error(f"Error getting districts info: {str(e)}")
            return []
    
    def validate_geojson_structure(self) -> bool:
        """Validate that the GeoJSON has required fields"""
        try:
            gdf = self.load_kecamatan_geojson()
            if gdf is None:
                return False
            
            required_fields = ['GID_3', 'NAME_3', 'NAME_2', 'nasa_match', 'nasa_lat', 'nasa_lng']
            
            for field in required_fields:
                if field not in gdf.columns:
                    logger.error(f"Required field missing: {field}")
                    return False
            
            logger.info("GeoJSON structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating GeoJSON structure: {str(e)}")
            return False
    
    def get_nasa_matches(self) -> List[str]:
        """Get list of all NASA location matches"""
        try:
            gdf = self.load_kecamatan_geojson()
            if gdf is None:
                return []
            
            nasa_matches = gdf['nasa_match'].unique().tolist()
            logger.info(f"📋 Found {len(nasa_matches)} unique NASA matches")
            return nasa_matches
            
        except Exception as e:
            logger.error(f"Error getting NASA matches: {str(e)}")
            return []
        