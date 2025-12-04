import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from services.area_weight_service import AreaWeightService, KecamatanAreaInfo

logger = logging.getLogger(__name__)

@dataclass
class KecamatanInfo:
    """Complete kecamatan information with administrative and geographic data"""
    kecamatan_name: str
    kecamatan_code: str
    kabupaten_name: str
    kabupaten_code: str
    province: str
    area_km2: float
    area_weight: float
    nasa_location_name: Optional[str] = None  # Original NASA location name if different

@dataclass
class KabupatenInfo:
    """Kabupaten information with constituent kecamatan"""
    kabupaten_name: str
    kabupaten_code: str
    province: str
    total_area_km2: float
    kecamatan_count: int
    constituent_kecamatan: List[str]
    bps_compatible_name: str

class AdministrativeLevel(Enum):
    """Administrative level classification"""
    KECAMATAN = "kecamatan"
    KABUPATEN = "kabupaten"
    PROVINSI = "provinsi"

class KecamatanKabupatenMappingService:
    """
    Service for managing kecamatan-kabupaten administrative mapping
    Integrates with AreaWeightService for accurate geographic data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize area weight service with actual GeoJSON data
        self.area_service = AreaWeightService()
        
        # Administrative codes mapping
        self.kabupaten_codes = {
            "AcehBesar": "1171",
            "AcehJaya": "1118", 
            "AcehUtara": "1104",
            "Bireuen": "1111",
            "Pidie": "1103"
        }
        
        # NASA location to kecamatan name mapping
        # Based on your test results and nasa_match field
        self.nasa_to_kecamatan_mapping = {
            # NASA Location → Actual Kecamatan Name
            "Indrapuri": "Indrapuri",
            "Montasik": "Montasik", 
            "Darussalam": "Darussalam",
            "Setia Bakti": "Sampoiniet",
            "Teunom": "Teunom",
            "Pidie": "Grong-Grong",
            "Indrajaya": "Delima",
            "Lhoksukon": "Lhoksukon",
            "Juli": "Peudada",
            "Kota Juang": "Jeumpa"
        }
        
        # Reverse mapping for lookup
        self.kecamatan_to_nasa_mapping = {v: k for k, v in self.nasa_to_kecamatan_mapping.items()}
        
        # BPS API compatible names
        self.bps_kabupaten_mapping = {
            "AcehBesar": "Aceh Besar",
            "AcehJaya": "Aceh Jaya", 
            "AcehUtara": "Aceh Utara",
            "Bireuen": "Bireuen",
            "Pidie": "Pidie"
        }
        
        self.logger.info("Kecamatan-Kabupaten Mapping Service initialized with GeoJSON area data")
    
    def get_kecamatan_info(self, identifier: str) -> Optional[KecamatanInfo]:
        """
        Get complete kecamatan information
        
        Args:
            identifier: Can be NASA location name or kecamatan name
            
        Returns:
            KecamatanInfo object or None if not found
        """
        # Handle NASA location names
        actual_kecamatan = self.nasa_to_kecamatan_mapping.get(identifier, identifier)
        
        # Get area info from GeoJSON-based service
        area_info = self.area_service.get_kecamatan_area_info(actual_kecamatan)
        
        if not area_info:
            self.logger.warning(f"No area info found for kecamatan: {actual_kecamatan}")
            return None
        
        # Get kabupaten code
        kabupaten_code = self.kabupaten_codes.get(area_info.kabupaten_name)
        if not kabupaten_code:
            self.logger.warning(f"No kabupaten code found for: {area_info.kabupaten_name}")
            return None
        
        # Generate kecamatan code (kabupaten_code + unique suffix)
        kecamatan_code = f"{kabupaten_code}{str(hash(actual_kecamatan))[-3:]:0>3}"
        
        # Get NASA location name if different
        nasa_location = self.kecamatan_to_nasa_mapping.get(actual_kecamatan)
        
        return KecamatanInfo(
            kecamatan_name=actual_kecamatan,
            kecamatan_code=kecamatan_code,
            kabupaten_name=area_info.kabupaten_name,
            kabupaten_code=kabupaten_code,
            province="Aceh",
            area_km2=area_info.area_km2,
            area_weight=area_info.area_weight,
            nasa_location_name=nasa_location if nasa_location != actual_kecamatan else None
        )
    
    def get_kabupaten_info(self, kabupaten_name: str) -> Optional[KabupatenInfo]:
        """Get complete kabupaten information"""
        # Get constituent kecamatan
        constituent_kecamatan = self.area_service.get_kecamatan_by_kabupaten(kabupaten_name)
        
        if not constituent_kecamatan:
            return None
        
        # Get total area
        total_area = self.area_service.get_kabupaten_total_area(kabupaten_name)
        
        # Get codes and BPS name
        kabupaten_code = self.kabupaten_codes.get(kabupaten_name, "Unknown")
        bps_name = self.bps_kabupaten_mapping.get(kabupaten_name, kabupaten_name)
        
        return KabupatenInfo(
            kabupaten_name=kabupaten_name,
            kabupaten_code=kabupaten_code,
            province="Aceh",
            total_area_km2=total_area,
            kecamatan_count=len(constituent_kecamatan),
            constituent_kecamatan=constituent_kecamatan,
            bps_compatible_name=bps_name
        )
    
    def get_kabupaten_from_kecamatan(self, identifier: str) -> Optional[str]:
        """
        Get kabupaten name from kecamatan identifier
        
        Args:
            identifier: NASA location name or kecamatan name
        """
        kecamatan_info = self.get_kecamatan_info(identifier)
        return kecamatan_info.kabupaten_name if kecamatan_info else None
    
    def get_kecamatan_by_kabupaten(self, kabupaten_name: str) -> List[str]:
        """Get all kecamatan names within a kabupaten"""
        return self.area_service.get_kecamatan_by_kabupaten(kabupaten_name)
    
    def get_area_weights_for_kabupaten(self, kabupaten_name: str) -> Dict[str, float]:
        """Get area weights for all kecamatan within kabupaten"""
        return self.area_service.get_area_weights_for_kabupaten(kabupaten_name)
    
    def get_all_kabupaten(self) -> List[str]:
        """Get all kabupaten names"""
        return self.area_service.get_all_kabupaten()
    
    def get_all_kecamatan(self) -> List[str]:
        """Get all kecamatan names"""
        return list(self.area_service.kecamatan_info.keys())
    
    def get_all_nasa_locations(self) -> List[str]:
        """Get all NASA location names"""
        return list(self.nasa_to_kecamatan_mapping.keys())
    
    def get_bps_compatible_kabupaten_names(self) -> Dict[str, str]:
        """Get mapping from internal kabupaten names to BPS API names"""
        return self.bps_kabupaten_mapping.copy()
    
    def validate_mapping_consistency(self) -> Dict[str, Any]:
        """Validate mapping consistency across all data sources"""
        validation_results = {
            "area_service_status": "connected",
            "total_kabupaten": len(self.get_all_kabupaten()),
            "total_kecamatan": len(self.get_all_kecamatan()),
            "total_nasa_locations": len(self.get_all_nasa_locations()),
            "kabupaten_validation": {},
            "nasa_mapping_validation": {},
            "bps_mapping_validation": {},
            "administrative_codes_validation": {}
        }
        
        # Validate each kabupaten
        for kabupaten_name in self.get_all_kabupaten():
            kabupaten_info = self.get_kabupaten_info(kabupaten_name)
            area_weights = self.get_area_weights_for_kabupaten(kabupaten_name)
            
            validation_results["kabupaten_validation"][kabupaten_name] = {
                "kecamatan_count": len(area_weights),
                "constituent_kecamatan": kabupaten_info.constituent_kecamatan if kabupaten_info else [],
                "total_area_km2": kabupaten_info.total_area_km2 if kabupaten_info else 0,
                "weight_sum": round(sum(area_weights.values()), 4),
                "weight_sum_valid": abs(sum(area_weights.values()) - 1.0) < 0.001,
                "kabupaten_code": kabupaten_info.kabupaten_code if kabupaten_info else None,
                "bps_compatible_name": kabupaten_info.bps_compatible_name if kabupaten_info else None
            }
        
        # Validate NASA location mapping
        for nasa_location, kecamatan_name in self.nasa_to_kecamatan_mapping.items():
            kecamatan_info = self.get_kecamatan_info(nasa_location)
            
            validation_results["nasa_mapping_validation"][nasa_location] = {
                "maps_to_kecamatan": kecamatan_name,
                "kecamatan_exists": kecamatan_info is not None,
                "kabupaten": kecamatan_info.kabupaten_name if kecamatan_info else None,
                "area_km2": kecamatan_info.area_km2 if kecamatan_info else None,
                "area_weight": kecamatan_info.area_weight if kecamatan_info else None,
                "kecamatan_code": kecamatan_info.kecamatan_code if kecamatan_info else None
            }
        
        # Validate BPS mapping
        for internal_name, bps_name in self.bps_kabupaten_mapping.items():
            validation_results["bps_mapping_validation"][internal_name] = {
                "bps_compatible_name": bps_name,
                "kabupaten_exists": internal_name in self.get_all_kabupaten(),
                "has_kecamatan": len(self.get_kecamatan_by_kabupaten(internal_name)) > 0
            }
        
        # Validate administrative codes
        for kabupaten_name, code in self.kabupaten_codes.items():
            validation_results["administrative_codes_validation"][kabupaten_name] = {
                "kabupaten_code": code,
                "code_format_valid": len(code) == 4 and code.isdigit(),
                "kabupaten_exists": kabupaten_name in self.get_all_kabupaten()
            }
        
        self.logger.info("Mapping consistency validation completed")
        return validation_results
    
    def get_detailed_mapping_summary(self) -> Dict[str, Any]:
        """Get detailed summary of all mappings"""
        summary = {
            "service_info": {
                "total_kabupaten": len(self.get_all_kabupaten()),
                "total_kecamatan": len(self.get_all_kecamatan()),
                "total_nasa_locations": len(self.get_all_nasa_locations()),
                "area_data_source": self.area_service.geojson_path
            },
            "kabupaten_details": {},
            "nasa_location_mapping": {},
            "administrative_hierarchy": {}
        }
        
        # Detailed kabupaten information
        for kabupaten_name in self.get_all_kabupaten():
            kabupaten_info = self.get_kabupaten_info(kabupaten_name)
            area_weights = self.get_area_weights_for_kabupaten(kabupaten_name)
            
            kecamatan_details = []
            for kecamatan_name in kabupaten_info.constituent_kecamatan:
                kecamatan_info = self.get_kecamatan_info(kecamatan_name)
                if kecamatan_info:
                    kecamatan_details.append({
                        "kecamatan_name": kecamatan_name,
                        "kecamatan_code": kecamatan_info.kecamatan_code,
                        "area_km2": kecamatan_info.area_km2,
                        "area_weight": kecamatan_info.area_weight,
                        "area_percentage": round(kecamatan_info.area_weight * 100, 1),
                        "nasa_location_name": kecamatan_info.nasa_location_name
                    })
            
            summary["kabupaten_details"][kabupaten_name] = {
                "kabupaten_code": kabupaten_info.kabupaten_code,
                "bps_compatible_name": kabupaten_info.bps_compatible_name,
                "total_area_km2": kabupaten_info.total_area_km2,
                "kecamatan_count": kabupaten_info.kecamatan_count,
                "kecamatan_details": sorted(kecamatan_details, key=lambda x: x["area_weight"], reverse=True)
            }
        
        # NASA location mapping details
        for nasa_location in self.get_all_nasa_locations():
            kecamatan_info = self.get_kecamatan_info(nasa_location)
            if kecamatan_info:
                summary["nasa_location_mapping"][nasa_location] = {
                    "kecamatan_name": kecamatan_info.kecamatan_name,
                    "kabupaten_name": kecamatan_info.kabupaten_name,
                    "area_km2": kecamatan_info.area_km2,
                    "area_weight": kecamatan_info.area_weight,
                    "kecamatan_code": kecamatan_info.kecamatan_code
                }
        
        # Administrative hierarchy
        summary["administrative_hierarchy"] = {
            "province": "Aceh",
            "kabupaten_structure": {}
        }
        
        for kabupaten_name in self.get_all_kabupaten():
            kecamatan_list = self.get_kecamatan_by_kabupaten(kabupaten_name)
            summary["administrative_hierarchy"]["kabupaten_structure"][kabupaten_name] = {
                "kecamatan_list": kecamatan_list,
                "nasa_locations_in_kabupaten": [
                    nasa_loc for nasa_loc, kec_name in self.nasa_to_kecamatan_mapping.items()
                    if kec_name in kecamatan_list
                ]
            }
        
        return summary
    
    def get_aggregation_weights_for_analysis(self, kabupaten_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregation weights formatted for two-level analysis
        
        Returns weights and metadata needed for FSCI aggregation
        """
        weights = self.get_area_weights_for_kabupaten(kabupaten_name)
        kabupaten_info = self.get_kabupaten_info(kabupaten_name)
        
        aggregation_data = {
            "kabupaten_info": {
                "name": kabupaten_name,
                "bps_name": kabupaten_info.bps_compatible_name if kabupaten_info else kabupaten_name,
                "total_area_km2": kabupaten_info.total_area_km2 if kabupaten_info else 0,
                "kecamatan_count": kabupaten_info.kecamatan_count if kabupaten_info else 0
            },
            "kecamatan_weights": {},
            "nasa_location_weights": {}
        }
        
        # Kecamatan weights
        for kecamatan_name, weight in weights.items():
            kecamatan_info = self.get_kecamatan_info(kecamatan_name)
            if kecamatan_info:
                aggregation_data["kecamatan_weights"][kecamatan_name] = {
                    "weight": weight,
                    "area_km2": kecamatan_info.area_km2,
                    "kecamatan_code": kecamatan_info.kecamatan_code
                }
                
                # NASA location weights (for mapping from NASA location results)
                nasa_location = kecamatan_info.nasa_location_name or kecamatan_name
                aggregation_data["nasa_location_weights"][nasa_location] = {
                    "weight": weight,
                    "kecamatan_name": kecamatan_name,
                    "area_km2": kecamatan_info.area_km2
                }
        
        return aggregation_data