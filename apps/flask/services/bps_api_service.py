import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class RiceProductionData:
    kabupaten: str
    kode_wilayah: int
    produksi_padi_ton: float
    produksi_beras_ton: float
    year: int

class BPSApiService:
    """
    Service to fetch rice production data from BPS Indonesia API
    """
    
    def __init__(self):
        self.base_url = "https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi"
        self.api_key = "d83593d2486e73d9e28f059008bcfdcc"
        self.province_code = "1100000"  # Aceh Province
        self.table_id = "d3ZjM280TU9FanlkdDRETUV5aVdndz09"  # Rice production table
        
        # Target kabupaten for spatial analysis
        self.target_kabupaten = [
            "Aceh Besar",
            "Aceh Jaya", 
            "Pidie",
            "Aceh Utara",
            "Bireuen"
        ]
        
        # Mapping for consistent naming with spatial data
        self.kabupaten_mapping = {
            "Aceh Besar": "Aceh Besar",
            "Aceh Jaya": "Aceh Jaya",
            "Pidie": "Pidie", 
            "Aceh Utara": "Aceh Utara",
            "Bireuen": "Bireuen"
        }
        
        self.logger = logging.getLogger(__name__)
    
    def fetch_rice_production_data(self, year: int = 2024) -> List[RiceProductionData]:
        """
        Fetch rice production data for target kabupaten from BPS API
        
        Args:
            year: Year of data to fetch (default: 2024)
            
        Returns:
            List of RiceProductionData for target kabupaten
        """
        self.logger.info(f"Fetching BPS rice production data for year {year}")
        
        try:
            # Construct API URL
            url = f"{self.base_url}/id/25/tahun/{year}/id_tabel/{self.table_id}/wilayah/{self.province_code}/key/{self.api_key}"
            
            # Make API request
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK':
                raise ValueError(f"BPS API returned status: {data.get('status')}")
                
            if data.get('data-availability') != 'available':
                raise ValueError("BPS data not available for requested parameters")
                
            # Extract production data
            production_records = []
            raw_data = data['data'][1]['data']  # Second element contains the actual data
            
            self.logger.info(f"Processing {len(raw_data)} kabupaten records from BPS")
            
            for record in raw_data:
                kabupaten_name = record['label']
                
                # Filter only target kabupaten
                if kabupaten_name in self.target_kabupaten:
                    try:
                        production_data = self._parse_production_record(record, year)
                        if production_data:
                            production_records.append(production_data)
                            self.logger.info(f"✅ {kabupaten_name}: {production_data.produksi_padi_ton:.2f} ton padi")
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing data for {kabupaten_name}: {str(e)}")
                        continue
            
            self.logger.info(f"Successfully fetched production data for {len(production_records)} kabupaten")
            return production_records
            
        except requests.RequestException as e:
            self.logger.error(f"HTTP error fetching BPS data: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing BPS data: {str(e)}")
            raise
    
    def fetch_multi_year_production_data(self, start_year: int = 2018, end_year: int = 2024) -> Dict[int, List[RiceProductionData]]:
        """
        Fetch rice production data for multiple years
        
        Args:
            start_year: Starting year (default: 2018)
            end_year: Ending year (default: 2024)
            
        Returns:
            Dictionary mapping year -> List of RiceProductionData
        """
        self.logger.info(f"Fetching multi-year BPS data from {start_year} to {end_year}")
        
        multi_year_data = {}
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            try:
                self.logger.info(f"Fetching data for year {year}...")
                year_data = self.fetch_rice_production_data(year)
                
                if year_data:
                    multi_year_data[year] = year_data
                    self.logger.info(f"✅ Successfully fetched {len(year_data)} records for {year}")
                else:
                    failed_years.append(year)
                    self.logger.warning(f"❌ No data available for {year}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch data for {year}: {str(e)}")
                failed_years.append(year)
                continue
        
        self.logger.info(f"Multi-year fetch complete: {len(multi_year_data)} years successful, {len(failed_years)} failed")
        
        if failed_years:
            self.logger.warning(f"Failed years: {failed_years}")
        
        return multi_year_data
    
    def _parse_production_record(self, record: Dict[str, Any], year: int) -> Optional[RiceProductionData]:
        """
        Parse individual kabupaten production record from BPS response
        
        Args:
            record: Raw record from BPS API
            year: Year of the data
            
        Returns:
            RiceProductionData object or None if parsing fails
        """
        try:
            kabupaten = record['label']
            kode_wilayah = record['kode_wilayah']
            variables = record['variables']
            
            # Extract production values (handle Indonesian number format)
            # zuxztj3b0i = Produksi Padi (ton)
            # jufsvcze9h = Produksi Beras (ton)
            
            padi_raw = variables.get('zuxztj3b0i', {}).get('value_raw')
            beras_raw = variables.get('jufsvcze9h', {}).get('value_raw')
            
            if not padi_raw or not beras_raw:
                self.logger.warning(f"Missing production data for {kabupaten}")
                return None
            
            # Convert Indonesian number format to float
            # "178.318,81" -> 178318.81
            produksi_padi = self._parse_indonesian_number(padi_raw)
            produksi_beras = self._parse_indonesian_number(beras_raw)
            
            if produksi_padi is None or produksi_beras is None:
                self.logger.warning(f"Could not parse numbers for {kabupaten}: padi={padi_raw}, beras={beras_raw}")
                return None
            
            return RiceProductionData(
                kabupaten=kabupaten,
                kode_wilayah=kode_wilayah,
                produksi_padi_ton=produksi_padi,
                produksi_beras_ton=produksi_beras,
                year=year
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing production record: {str(e)}")
            return None
    
    def _parse_indonesian_number(self, number_str: str) -> Optional[float]:
        """
        Parse Indonesian number format (dot as thousand separator, comma as decimal)
        
        Args:
            number_str: Indonesian formatted number string (e.g., "178.318,81")
            
        Returns:
            Float value or None if parsing fails
        """
        if not number_str or number_str in ['–', '...', 'NA']:
            return None
            
        try:
            # Handle Indonesian number format: "178.318,81" -> 178318.81
            # Remove thousand separators (dots)
            cleaned = number_str.replace('.', '')
            # Replace decimal separator (comma) with dot
            cleaned = cleaned.replace(',', '.')
            
            return float(cleaned)
            
        except (ValueError, AttributeError):
            return None
    
    def fetch_kabupaten_historical_data(
        self,
        kabupaten_name: str,
        start_year: int = 2018,
        end_year: int = 2024
    ) -> Dict[int, RiceProductionData]:
        """
        Fetch historical production data for specific kabupaten
        
        Args:
            kabupaten_name: Name of kabupaten
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            Dictionary mapping year -> RiceProductionData for the kabupaten
        """
        self.logger.info(f"Fetching historical data for {kabupaten_name} ({start_year}-{end_year})")
        
        # Get multi-year data with minimal logging
        kabupaten_historical = {}
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            try:
                # Fetch single year data
                year_data = self.fetch_rice_production_data(year)
                
                # Find the specific kabupaten
                for record in year_data:
                    if record.kabupaten.lower() == kabupaten_name.lower():
                        kabupaten_historical[year] = record
                        break
                        
            except Exception as e:
                self.logger.warning(f"Failed to fetch {year} data: {str(e)}")
                failed_years.append(year)
                continue
        
        self.logger.info(f"Historical data for {kabupaten_name}: {len(kabupaten_historical)}/{end_year - start_year + 1} years")
        
        if failed_years:
            self.logger.warning(f"Missing years: {failed_years}")
        
        return kabupaten_historical
    
    def get_production_summary(self, year: int = 2024) -> Dict[str, Any]:
        """
        Get summary statistics for rice production in target kabupaten
        
        Args:
            year: Year of data to analyze
            
        Returns:
            Summary statistics dictionary
        """
        try:
            production_data = self.fetch_rice_production_data(year)
            
            if not production_data:
                return {"error": "No production data available"}
            
            total_padi = sum(record.produksi_padi_ton for record in production_data)
            total_beras = sum(record.produksi_beras_ton for record in production_data)
            
            # Calculate per-kabupaten statistics
            kabupaten_stats = []
            for record in production_data:
                kabupaten_stats.append({
                    "kabupaten": record.kabupaten,
                    "kode_wilayah": record.kode_wilayah,
                    "produksi_padi_ton": record.produksi_padi_ton,
                    "produksi_beras_ton": record.produksi_beras_ton,
                    "padi_percentage": round((record.produksi_padi_ton / total_padi) * 100, 2),
                    "beras_percentage": round((record.produksi_beras_ton / total_beras) * 100, 2),
                    "conversion_rate": round((record.produksi_beras_ton / record.produksi_padi_ton) * 100, 2) if record.produksi_padi_ton > 0 else 0
                })
            
            # Sort by production volume
            kabupaten_stats.sort(key=lambda x: x['produksi_padi_ton'], reverse=True)
            
            summary = {
                "year": year,
                "total_kabupaten": len(production_data),
                "aggregate_production": {
                    "total_padi_ton": round(total_padi, 2),
                    "total_beras_ton": round(total_beras, 2),
                    "average_padi_per_kabupaten": round(total_padi / len(production_data), 2),
                    "average_beras_per_kabupaten": round(total_beras / len(production_data), 2),
                    "overall_conversion_rate": round((total_beras / total_padi) * 100, 2) if total_padi > 0 else 0
                },
                "kabupaten_ranking": kabupaten_stats,
                "top_producer": kabupaten_stats[0] if kabupaten_stats else None,
                "data_source": "BPS Indonesia API",
                "fetch_timestamp": None  # Will be set by API endpoint
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating production summary: {str(e)}")
            return {"error": str(e)}
    
    def get_production_for_spatial_analysis(self, year: int = 2024) -> Dict[str, float]:
        """
        Get production data formatted for spatial analysis integration
        Returns simple kabupaten -> production mapping
        
        Args:
            year: Year of data to fetch
            
        Returns:
            Dictionary mapping kabupaten name to padi production (tons)
        """
        try:
            production_data = self.fetch_rice_production_data(year)
            
            # Create simple mapping for spatial integration
            production_mapping = {}
            for record in production_data:
                # Use standardized kabupaten name for spatial matching
                standardized_name = self.kabupaten_mapping.get(record.kabupaten, record.kabupaten)
                production_mapping[standardized_name] = record.produksi_padi_ton
                
            self.logger.info(f"Prepared production mapping for {len(production_mapping)} kabupaten")
            return production_mapping
            
        except Exception as e:
            self.logger.error(f"Error preparing production data for spatial analysis: {str(e)}")
            return {}