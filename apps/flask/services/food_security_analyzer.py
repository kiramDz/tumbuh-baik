import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class FSIClass(Enum):
    """Food Security Index Classification"""
    SANGAT_TINGGI = "Sangat Tinggi"    # FSI >= 80
    TINGGI = "Tinggi"                  # FSI 60-79
    SEDANG = "Sedang"                  # FSI 40-59
    RENDAH = "Rendah"                  # FSI < 40
    
    
@dataclass
class FoodSecurityAnalysis:
    """Food Security Index analysis"""
    district_name: str
    district_code: str
    fsi_score: float
    fsi_class: FSIClass
    natural_resources_score: float    # Component 1
    availability_score: float         # Component 2  
    analysis_timestamp: str
    

class FoodSecurityAnalyzer:
    """
    Component 1: Natural Resources & Resilience (climate sustainability)
    Component 2: Availability (food supply adequacy)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.fsi_weights = {
            'natural_resources': 0.60,  # Climate sustainability
            'availability': 0.40        # Food supply adequacy
        }
        
        self.logger.info("Food Security Index analyzer initialized (FSI)")
        
    def analyze_food_security(self, 
                            district_data: Dict[str, Any],
                            climate_time_series: List[Dict[str, Any]],
                            base_suitability_score: float) -> FoodSecurityAnalysis:
        """
        Food security analysis for a district
        
        Args:
            district_data: Spatial data for the district
            climate_time_series: Climate time series data
            base_suitability_score: Base climate suitability score
            
        Returns:
            FoodSecurityAnalysis object
        """
        try:
            district_name = district_data.get('NAME_3', district_data.get('NAME_2', 'Unknown District'))
            district_code = district_data.get('GID_3', district_data.get('GID_2', 'Unknown'))
            
            self.logger.info(f"Starting FSI analysis for {district_name}")
            
            # 1. Natural Resources & Resilience Score
            natural_resources_score = self._calculate_natural_resources_score(
                climate_time_series, base_suitability_score
            )
            
            # 2. Availability Score (climate-based proxy for now)
            availability_score = self._calculate_availability_score(
                climate_time_series, base_suitability_score
            )
            
            # 3. Food Security Index
            fsi_score = self._calculate_fsi_score(natural_resources_score, availability_score)
            fsi_class = self._classify_fsi_score(fsi_score)
            
            analysis = FoodSecurityAnalysis(
                district_name=district_name,
                district_code=district_code,
                fsi_score=fsi_score,
                fsi_class=fsi_class,
                natural_resources_score=natural_resources_score,
                availability_score=availability_score,
                analysis_timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"FSI analysis complete for {district_name}: "
                           f"FSI={fsi_score:.1f} ({fsi_class.value})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in FSI analysis: {str(e)}")
            raise
        
    def _calculate_natural_resources_score(self, 
                                           climate_data: List[Dict[str, Any]],
                                           base_suitability: float) -> float:
        """Calculate Natural Resources & Resilience Score"""
        
        try:
            if not climate_data:
                return base_suitability
            
            df = pd.DataFrame(climate_data)
            
            # Climate sustainability (60% weight)
            climate_sustainability = base_suitability
            
            # Climate stability (40% weight) - lower variability = better
            stability_score = 75.0  # default
            if 'T2M' in df.columns and 'PRECTOTCORR' in df.columns:
                temp_cv = (df['T2M'].std() / df['T2M'].mean()) * 100
                precip_cv = (df['PRECTOTCORR'].std() / df['PRECTOTCORR'].mean()) * 100 if df['PRECTOTCORR'].mean() > 0 else 30
                
                # Lower coefficient of variation = higher stability
                temp_stability = max(50, 100 - (temp_cv * 10))
                precip_stability = max(50, 100 - (precip_cv * 2))
                
                stability_score = (temp_stability + precip_stability) / 2
            
            # Combine sustainability and stability
            natural_resources = (climate_sustainability * 0.6) + (stability_score * 0.4)
            
            return min(100, max(0, natural_resources))
        except Exception as e:
            self.logger.error(f"Error calculating natural resources score: {str(e)}")
            return base_suitability * 0.8
        
    def _calculate_availability_score(self,
                                    climate_data: List[Dict[str, Any]],
                                    base_suitability: float) -> float:
        """Calculate Availability score (climate-based proxy)"""
        try:
            if not climate_data:
                return base_suitability * 0.85
            
            df = pd.DataFrame(climate_data)
            
            # Water availability for rice production
            water_score = 70.0  # default
            if 'PRECTOTCORR' in df.columns:
                avg_precip = df['PRECTOTCORR'].mean()
                annual_precip = avg_precip * 365
                
                # Optimal range: 1200-1800mm annually for rice
                if 1200 <= annual_precip <= 1800:
                    water_score = 100
                elif annual_precip > 1800:
                    water_score = max(60, 100 - ((annual_precip - 1800) / 1000) * 30)
                else:
                    water_score = max(40, (annual_precip / 1200) * 100)
            
            # Temperature suitability for rice growth
            temp_score = 75.0  # default
            if 'T2M' in df.columns:
                avg_temp = df['T2M'].mean()
                
                # Optimal range: 24-30°C for rice
                if 24 <= avg_temp <= 30:
                    temp_score = 100
                elif avg_temp > 30:
                    temp_score = max(60, 100 - ((avg_temp - 30) / 5) * 40)
                else:
                    temp_score = max(50, ((avg_temp - 20) / 4) * 100)
            
            # Combine water and temperature availability
            availability = (water_score * 0.6) + (temp_score * 0.4)
            
            return min(100, max(30, availability))
            
        except Exception as e:
            self.logger.error(f"Error calculating availability score: {str(e)}")
            return base_suitability * 0.75  # Conservative estimate
        
    def _calculate_fsi_score(self, 
                           natural_resources: float, 
                           availability: float) -> float:
        """Calculate Food Security Index score"""
        try:
            fsi = (
                natural_resources * self.fsi_weights['natural_resources'] +
                availability * self.fsi_weights['availability']
            )
            
            return round(fsi, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating FSI score: {str(e)}")
            return 65.0
    
    def _classify_fsi_score(self, score: float) -> FSIClass:
        """Classify FSI score"""
        if score >= 80:
            return FSIClass.SANGAT_TINGGI
        elif score >= 60:
            return FSIClass.TINGGI
        elif score >= 40:
            return FSIClass.SEDANG
        else:
            return FSIClass.RENDAH
            