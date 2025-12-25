import logging
import geopandas as gpd
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)
@dataclass
class KecamatanAreaInfo:
    """Area information for kecamatan"""
    kecamatan_name: str
    kabupaten_name: str
    area_km2: float
    area_weight: float  

class AreaWeightService:
    """
    Service for calculating accurate area weights from GeoJSON
    Calculates: area_weight = (kecamatan_area) ÷ (total_kabupaten_area)
    """
    
    def __init__(self, geojson_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default path to your GeoJSON file
        if geojson_path is None:
            geojson_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/kkp-only/tumbuh-baik/apps/flask/data/geojson/aceh_nasa_kecamatan.geojson"
        
        self.geojson_path = geojson_path
        self.area_weights = None
        self.kecamatan_info = {}
        
        # Load and calculate area weights on initialization
        self._load_and_calculate_weights()
    
    def _load_and_calculate_weights(self):
        """Load GeoJSON and calculate area weights"""
        try:
            if not os.path.exists(self.geojson_path):
                self.logger.error(f"GeoJSON file not found: {self.geojson_path}")
                raise FileNotFoundError(f"GeoJSON file not found: {self.geojson_path}")
            
            self.logger.info(f"Loading GeoJSON from: {self.geojson_path}")
            
            # Load GeoJSON
            gdf = gpd.read_file(self.geojson_path)
            self.logger.info(f"Loaded {len(gdf)} kecamatan from GeoJSON")
            
            # Convert CRS to projected for accurate area calculation
            # Aceh ~ UTM zone 46N (EPSG:32646)
            if gdf.crs != 'EPSG:32646':
                gdf = gdf.to_crs(epsg=32646)
                self.logger.info("Converted CRS to UTM 46N for accurate area calculation")
            
            # Compute area in km²
            gdf["area_km2"] = gdf.geometry.area / 1_000_000
            
            # Group by kabupaten (using NAME_2 field)
            kabupaten_groups = gdf.groupby("NAME_2")
            
            self.area_weights = {}
            
            for kabupaten, group in kabupaten_groups:
                total_area = group["area_km2"].sum()
                
                kecamatan_weights = {}
                kabupaten_kecamatan_info = {}
                
                for _, row in group.iterrows():
                    kecamatan_name = row["NAME_3"]
                    kecamatan_area = row["area_km2"]
                    weight = kecamatan_area / total_area
                    
                    # Store weight
                    kecamatan_weights[kecamatan_name] = round(float(weight), 4)
                    
                    # Store detailed info
                    self.kecamatan_info[kecamatan_name] = KecamatanAreaInfo(
                        kecamatan_name=kecamatan_name,
                        kabupaten_name=kabupaten,
                        area_km2=round(float(kecamatan_area), 2),
                        area_weight=round(float(weight), 4)
                    )
                
                self.area_weights[kabupaten] = {
                    "total_area_km2": round(float(total_area), 2),
                    "kecamatan_count": len(group),
                    "kecamatan_weights": kecamatan_weights
                }
                
                self.logger.info(f"Calculated weights for {kabupaten}: "
                               f"{len(group)} kecamatan, {total_area:.1f} km²")
            
            self.logger.info(f"Area weights calculated for {len(self.area_weights)} kabupaten")
            
        except Exception as e:
            self.logger.error(f"Error loading GeoJSON and calculating weights: {str(e)}")
            raise
    
    def get_area_weight(self, kecamatan_name: str) -> Optional[float]:
        """Get area weight for specific kecamatan"""
        kecamatan_info = self.kecamatan_info.get(kecamatan_name)
        return kecamatan_info.area_weight if kecamatan_info else None
    
    def get_kecamatan_area_info(self, kecamatan_name: str) -> Optional[KecamatanAreaInfo]:
        """Get complete area information for kecamatan"""
        return self.kecamatan_info.get(kecamatan_name)
    
    def get_area_weights_for_kabupaten(self, kabupaten_name: str) -> Dict[str, float]:
        """Get area weights for all kecamatan within a kabupaten"""
        kabupaten_data = self.area_weights.get(kabupaten_name, {})
        return kabupaten_data.get("kecamatan_weights", {})
    
    def get_kabupaten_total_area(self, kabupaten_name: str) -> float:
        """Get total area for kabupaten"""
        kabupaten_data = self.area_weights.get(kabupaten_name, {})
        return kabupaten_data.get("total_area_km2", 0.0)
    
    def get_all_kabupaten(self) -> list[str]:
        """Get all kabupaten names"""
        return list(self.area_weights.keys()) if self.area_weights else []
    
    def get_kecamatan_by_kabupaten(self, kabupaten_name: str) -> list[str]:
        """Get all kecamatan names within kabupaten"""
        kabupaten_data = self.area_weights.get(kabupaten_name, {})
        kecamatan_weights = kabupaten_data.get("kecamatan_weights", {})
        return list(kecamatan_weights.keys())
    
    def validate_area_weights(self) -> Dict[str, Any]:
        """Validate area weight calculations"""
        validation_results = {
            "total_kabupaten": len(self.area_weights) if self.area_weights else 0,
            "total_kecamatan": len(self.kecamatan_info),
            "kabupaten_validation": {},
            "weight_sum_validation": {}
        }
        
        if not self.area_weights:
            validation_results["error"] = "No area weights available"
            return validation_results
        
        for kabupaten_name, kabupaten_data in self.area_weights.items():
            weights = kabupaten_data.get("kecamatan_weights", {})
            weight_sum = sum(weights.values())
            
            validation_results["kabupaten_validation"][kabupaten_name] = {
                "kecamatan_count": len(weights),
                "total_area_km2": kabupaten_data.get("total_area_km2", 0),
                "weight_sum": round(weight_sum, 4),
                "weight_sum_valid": abs(weight_sum - 1.0) < 0.001,  # Should sum to 1.0
                "kecamatan_weights": weights
            }
            
            validation_results["weight_sum_validation"][kabupaten_name] = {
                "expected": 1.0,
                "actual": round(weight_sum, 4),
                "difference": round(abs(weight_sum - 1.0), 4),
                "valid": abs(weight_sum - 1.0) < 0.001
            }
        
        return validation_results
    
    def get_area_weight_summary(self) -> Dict[str, Any]:
        """Get summary of all area weights"""
        if not self.area_weights:
            return {"error": "No area weights available"}
        
        summary = {
            "geojson_source": self.geojson_path,
            "total_kabupaten": len(self.area_weights),
            "total_kecamatan": len(self.kecamatan_info),
            "kabupaten_summary": {}
        }
        
        for kabupaten_name, kabupaten_data in self.area_weights.items():
            kecamatan_areas = []
            for kecamatan_name in kabupaten_data["kecamatan_weights"].keys():
                kecamatan_info = self.kecamatan_info.get(kecamatan_name)
                if kecamatan_info:
                    kecamatan_areas.append({
                        "kecamatan": kecamatan_name,
                        "area_km2": kecamatan_info.area_km2,
                        "weight": kecamatan_info.area_weight,
                        "percentage": round(kecamatan_info.area_weight * 100, 1)
                    })
            
            summary["kabupaten_summary"][kabupaten_name] = {
                "total_area_km2": kabupaten_data["total_area_km2"],
                "kecamatan_count": kabupaten_data["kecamatan_count"],
                "kecamatan_areas": sorted(kecamatan_areas, key=lambda x: x["weight"], reverse=True)
            }
        
        return summary
    
    def get_nasa_location_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get mapping between NASA locations and kecamatan with area weights"""
        if not self.area_weights:
            return {}
        
        # This maps your NASA location names to kecamatan names with area info
        nasa_mapping = {}
        
        # Map based on your GeoJSON nasa_match field
        nasa_location_map = {
            "Indrapuri": "Indrapuri",
            "Montasik": "Montasik", 
            "Darussalam": "Darussalam",
            "Setia Bakti": "Sampoiniet",  # nasa_match -> NAME_3
            "Teunom": "Teunom",
            "Pidie": "Grong-Grong",      # nasa_match -> NAME_3
            "Indrajaya": "Delima",       # nasa_match -> NAME_3
            "Lhoksukon": "Lhoksukon",
            "Juli": "Peudada",           # nasa_match -> NAME_3
            "Kota Juang": "Jeumpa"       # nasa_match -> NAME_3
        }
        
        for nasa_location, kecamatan_name in nasa_location_map.items():
            kecamatan_info = self.kecamatan_info.get(kecamatan_name)
            if kecamatan_info:
                nasa_mapping[nasa_location] = {
                    "kecamatan_name": kecamatan_name,
                    "kabupaten_name": kecamatan_info.kabupaten_name,
                    "area_km2": kecamatan_info.area_km2,
                    "area_weight": kecamatan_info.area_weight
                }
        
        return nasa_mapping