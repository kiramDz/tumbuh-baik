import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from services.temporal_analysis_service import TemporalAnalysisService

logger = logging.getLogger(__name__)

class FSCIClass(Enum):
    """Food Security Composite Index Classification"""
    LUMBUNG_PANGAN_PRIMER = "Lumbung Pangan Primer"      # FSCI >= 80
    LUMBUNG_PANGAN_SEKUNDER = "Lumbung Pangan Sekunder"  # FSCI 60-79
    ZONA_POTENSIAL = "Zona Potensial"                    # FSCI 40-59
    BUKAN_PRIORITAS = "Bukan Prioritas Lumbung Pangan"   # FSCI < 40

class PCIClass(Enum):
    """Production Capacity Index Classification"""
    SANGAT_TINGGI = "Sangat Tinggi"    # PCI >= 85
    TINGGI = "Tinggi"                  # PCI 70-84
    SEDANG = "Sedang"                  # PCI 55-69
    RENDAH = "Rendah"                  # PCI 40-54
    SANGAT_RENDAH = "Sangat Rendah"    # PCI < 40

class PSIClass(Enum):
    """Production Stability Index Classification"""
    SANGAT_STABIL = "Sangat Stabil"    # PSI >= 80
    STABIL = "Stabil"                  # PSI 65-79
    KURANG_STABIL = "Kurang Stabil"    # PSI 50-64
    TIDAK_STABIL = "Tidak Stabil"      # PSI < 50

class CRSClass(Enum):
    """Climate Resilience Score Classification"""
    SANGAT_RESILIEN = "Sangat Resilien"  # CRS >= 75
    RESILIEN = "Resilien"                # CRS 60-74
    KURANG_RESILIEN = "Kurang Resilien"  # CRS 45-59
    TIDAK_RESILIEN = "Tidak Resilien"    # CRS < 45

@dataclass
class ProductionCapacityIndex:
    """Production Capacity Index calculation results (without BPS for now)"""
    climate_suitability: float
    land_quality_factor: float
    water_availability_factor: float
    risk_adjustment_factor: float
    pci_score: float
    pci_class: PCIClass

@dataclass
class ProductionStabilityIndex:
    """Production Stability Index calculation results"""
    temporal_stability: float
    climate_variability: float
    trend_consistency: float
    anomaly_resilience: float
    psi_score: float
    psi_class: PSIClass

@dataclass
class ClimateResilienceScore:
    """Climate Resilience Score calculation results"""
    temperature_resilience: float
    precipitation_resilience: float
    extreme_weather_resilience: float
    adaptation_capacity: float
    crs_score: float
    crs_class: CRSClass

@dataclass
class FoodSecurityAnalysis:
    """Complete Food Security Composite Index analysis"""
    district_name: str
    district_code: str
    pci: ProductionCapacityIndex
    psi: ProductionStabilityIndex
    crs: ClimateResilienceScore
    fsci_score: float
    fsci_class: FSCIClass
    investment_recommendation: str
    priority_ranking: int
    analysis_timestamp: str

class FoodSecurityAnalyzer:
    """
    Comprehensive Food Security Analyzer (Phase 1 - without BPS integration)
    Combines climate suitability, temporal stability, and resilience metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temporal_service = TemporalAnalysisService()
        
        # FSCI component weights (Phase 1 without BPS)
        self.fsci_weights = {
            'pci': 0.40,  # Production capacity based on climate
            'psi': 0.35,  # Production stability from temporal analysis
            'crs': 0.25   # Climate resilience
        }
        
        self.logger.info("Food Security Analyzer initialized (Phase 1 - Climate Based)")
    
    def analyze_food_security(self, 
                            district_data: Dict[str, Any],
                            climate_time_series: List[Dict[str, Any]],
                            base_suitability_score: float) -> FoodSecurityAnalysis:
        """
        Complete food security analysis for a district (Phase 1)
        
        Args:
            district_data: Spatial data for the district
            climate_time_series: 20-year climate time series data
            base_suitability_score: Base climate suitability score
            
        Returns:
            Complete FoodSecurityAnalysis object
        """
        try:
            district_name = district_data.get('NAME_3', district_data.get('NAME_2', 'Unknown District'))
            district_code = district_data.get('GID_3', district_data.get('GID_2', 'Unknown'))
            
            self.logger.info(f"Starting food security analysis for {district_name}")
            
            # 1. Production Capacity Index (climate-based)
            pci = self._calculate_production_capacity_index(
                district_data, climate_time_series, base_suitability_score
            )
            
            # 2. Production Stability Index
            psi = self._calculate_production_stability_index(
                district_data, climate_time_series
            )
            
            # 3. Climate Resilience Score
            crs = self._calculate_climate_resilience_score(
                district_data, climate_time_series
            )
            
            # 4. Food Security Composite Index
            fsci_score = self._calculate_fsci_score(pci, psi, crs)
            fsci_class = self._classify_fsci_score(fsci_score)
            
            # 5. Investment recommendation
            investment_recommendation = self._generate_investment_recommendation(
                pci, psi, crs, fsci_score
            )
            
            analysis = FoodSecurityAnalysis(
                district_name=district_name,
                district_code=district_code,
                pci=pci,
                psi=psi,
                crs=crs,
                fsci_score=fsci_score,
                fsci_class=fsci_class,
                investment_recommendation=investment_recommendation,
                priority_ranking=0,  # Will be set when ranking multiple districts
                analysis_timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Food security analysis complete for {district_name}: "
                           f"FSCI={fsci_score:.1f} ({fsci_class.value})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in food security analysis: {str(e)}")
            raise
    
    def _calculate_production_capacity_index(self, 
                                           district_data: Dict[str, Any],
                                           climate_data: List[Dict[str, Any]], 
                                           base_suitability: float) -> ProductionCapacityIndex:
        """Calculate Production Capacity Index (climate-based for Phase 1)"""
        try:
            # 1. Climate Suitability (from rice analyzer)
            climate_suitability = base_suitability
            
            # 2. Land Quality Factor (based on climate proxies)
            land_quality_factor = self._calculate_land_quality_factor(climate_data)
            
            # 3. Water Availability Factor
            water_availability = self._calculate_water_availability(climate_data)
            
            # 4. Risk Adjustment Factor
            risk_factor = self._calculate_risk_factors(climate_data)
            
            # Calculate PCI score (climate-based formula)
            pci_score = (
                climate_suitability * 0.4 +
                land_quality_factor * 0.3 +
                water_availability * 0.2 +
                (100 - risk_factor) * 0.1
            )
            
            pci_score = min(100, max(0, pci_score))
            pci_class = self._classify_pci_score(pci_score)
            
            return ProductionCapacityIndex(
                climate_suitability=climate_suitability,
                land_quality_factor=land_quality_factor,
                water_availability_factor=water_availability,
                risk_adjustment_factor=risk_factor,
                pci_score=round(pci_score, 2),
                pci_class=pci_class
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating PCI: {str(e)}")
            raise
    
    def _calculate_land_quality_factor(self, climate_data: List[Dict[str, Any]]) -> float:
        """Calculate land quality factor based on climate indicators"""
        try:
            if not climate_data:
                return 75.0
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(climate_data)
            
            # Soil moisture proxy (from precipitation and humidity)
            avg_precip = df['PRECTOTCORR'].mean() if 'PRECTOTCORR' in df.columns else 0
            avg_humidity = df['RH2M'].mean() if 'RH2M' in df.columns else 70
            
            # Land quality scoring based on climate proxies
            # Optimal annual precipitation: 1200-1800mm for rice
            annual_precip = avg_precip * 365
            if 1200 <= annual_precip <= 1800:
                moisture_score = 100
            elif annual_precip > 1800:
                moisture_score = max(60, 100 - ((annual_precip - 1800) / 1000) * 30)
            else:
                moisture_score = max(40, (annual_precip / 1200) * 100)
            
            # Humidity score (optimal 70-85%)
            if 70 <= avg_humidity <= 85:
                humidity_score = 100
            elif avg_humidity > 85:
                humidity_score = max(70, 100 - ((avg_humidity - 85) / 15) * 30)
            else:
                humidity_score = max(50, (avg_humidity / 70) * 100)
            
            land_quality = (moisture_score * 0.6) + (humidity_score * 0.4)
            return min(100, max(40, land_quality))
            
        except Exception as e:
            self.logger.error(f"Error calculating land quality factor: {str(e)}")
            return 75.0
    
    def _calculate_water_availability(self, climate_data: List[Dict[str, Any]]) -> float:
        """Calculate water availability factor"""
        try:
            if not climate_data:
                return 70.0
            
            df = pd.DataFrame(climate_data)
            
            # Annual precipitation
            avg_precip = df['PRECTOTCORR'].mean() if 'PRECTOTCORR' in df.columns else 0
            annual_precip = avg_precip * 365
            
            # Precipitation distribution (less variability = better water availability)
            precip_cv = (df['PRECTOTCORR'].std() / df['PRECTOTCORR'].mean()) * 100 if 'PRECTOTCORR' in df.columns and df['PRECTOTCORR'].mean() > 0 else 50
            
            # Optimal range: 1000-2000mm annually for rice
            if 1000 <= annual_precip <= 2000:
                amount_score = 100
            elif annual_precip > 2000:
                amount_score = max(60, 100 - ((annual_precip - 2000) / 1000) * 20)
            else:
                amount_score = max(30, (annual_precip / 1000) * 100)
            
            # Distribution score (lower CV = more consistent water supply)
            distribution_score = max(50, 100 - (precip_cv - 30))  # 30% CV as baseline
            
            water_score = (amount_score * 0.7) + (distribution_score * 0.3)
            return min(100, water_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating water availability: {str(e)}")
            return 70.0
    
    def _calculate_risk_factors(self, climate_data: List[Dict[str, Any]]) -> float:
        """Calculate risk adjustment factor (higher values = higher risk)"""
        try:
            if not climate_data:
                return 30.0  # Moderate risk as default
            
            df = pd.DataFrame(climate_data)
            
            # Temperature risk (extreme temperatures and variability)
            if 'T2M' in df.columns:
                temp_mean = df['T2M'].mean()
                temp_std = df['T2M'].std()
                
                # Risk from extreme temperatures (outside 24-32°C range for rice)
                temp_extremes = len(df[(df['T2M'] < 20) | (df['T2M'] > 35)]) / len(df) * 100
                temp_risk = min(50, temp_extremes * 2)
                
                # Risk from high variability
                variability_risk = min(30, temp_std * 5)
            else:
                temp_risk = 25.0
                variability_risk = 15.0
            
            # Precipitation risk (drought and flood potential)
            if 'PRECTOTCORR' in df.columns:
                precip_cv = (df['PRECTOTCORR'].std() / df['PRECTOTCORR'].mean()) * 100 if df['PRECTOTCORR'].mean() > 0 else 30
                
                # Risk from high precipitation variability
                precip_risk = min(40, max(0, precip_cv - 20))  # Risk increases above 20% CV
                
                # Drought risk (consecutive low precipitation periods)
                drought_risk = self._calculate_drought_risk(df['PRECTOTCORR'])
            else:
                precip_risk = 20.0
                drought_risk = 15.0
            
            # Combined risk score
            total_risk = (temp_risk * 0.3) + (variability_risk * 0.2) + (precip_risk * 0.3) + (drought_risk * 0.2)
            
            return min(80, max(10, total_risk))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk factors: {str(e)}")
            return 30.0
    
    def _calculate_drought_risk(self, precip_series: pd.Series) -> float:
        """Calculate drought risk from precipitation data"""
        try:
            # Define drought threshold (e.g., below 25th percentile for extended periods)
            drought_threshold = precip_series.quantile(0.25)
            
            # Count consecutive dry periods
            dry_periods = []
            current_dry_length = 0
            
            for value in precip_series:
                if value < drought_threshold:
                    current_dry_length += 1
                else:
                    if current_dry_length > 0:
                        dry_periods.append(current_dry_length)
                        current_dry_length = 0
            
            # Add final period if it ends in drought
            if current_dry_length > 0:
                dry_periods.append(current_dry_length)
            
            # Risk based on frequency and length of dry periods
            if not dry_periods:
                return 5.0  # Very low drought risk
            
            avg_dry_length = np.mean(dry_periods)
            max_dry_length = max(dry_periods)
            
            # Risk increases with longer dry periods
            drought_risk = min(40, (avg_dry_length / 30) * 20 + (max_dry_length / 60) * 20)
            
            return drought_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating drought risk: {str(e)}")
            return 20.0
    
    def _calculate_production_stability_index(self, 
                                            district_data: Dict[str, Any],
                                            climate_data: List[Dict[str, Any]]) -> ProductionStabilityIndex:
        """Calculate Production Stability Index using temporal analysis"""
        try:
            # Use temporal analysis service for detailed stability metrics
            stability_results = self.temporal_service.analyze_temporal_stability(climate_data)
            
            # Extract stability components
            temporal_stability = stability_results.get('overall_stability', 75.0)
            climate_variability = stability_results.get('climate_variability', 25.0)
            trend_consistency = stability_results.get('trend_consistency', 70.0)
            anomaly_resilience = stability_results.get('anomaly_resilience', 75.0)
            
            # Calculate PSI score
            psi_score = (
                temporal_stability * 0.35 +
                (100 - climate_variability) * 0.25 +  # Lower variability = higher stability
                trend_consistency * 0.25 +
                anomaly_resilience * 0.15
            )
            
            psi_class = self._classify_psi_score(psi_score)
            
            return ProductionStabilityIndex(
                temporal_stability=temporal_stability,
                climate_variability=climate_variability,
                trend_consistency=trend_consistency,
                anomaly_resilience=anomaly_resilience,
                psi_score=round(psi_score, 2),
                psi_class=psi_class
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {str(e)}")
            # Return default PSI on error
            return ProductionStabilityIndex(
                temporal_stability=75.0,
                climate_variability=25.0,
                trend_consistency=70.0,
                anomaly_resilience=75.0,
                psi_score=72.5,
                psi_class=PSIClass.STABIL
            )
    
    def _calculate_climate_resilience_score(self, 
                                          district_data: Dict[str, Any],
                                          climate_data: List[Dict[str, Any]]) -> ClimateResilienceScore:
        """Calculate Climate Resilience Score"""
        try:
            df = pd.DataFrame(climate_data)
            
            # Temperature resilience (lower variability = higher resilience)
            if 'T2M' in df.columns:
                temp_cv = (df['T2M'].std() / df['T2M'].mean()) * 100
                temperature_resilience = max(40, 100 - (temp_cv * 8))
            else:
                temperature_resilience = 70.0
            
            # Precipitation resilience
            if 'PRECTOTCORR' in df.columns:
                precip_cv = (df['PRECTOTCORR'].std() / df['PRECTOTCORR'].mean()) * 100 if df['PRECTOTCORR'].mean() > 0 else 30
                precipitation_resilience = max(40, 100 - (precip_cv * 1.5))
            else:
                precipitation_resilience = 70.0
            
            # Extreme weather resilience (based on outliers frequency)
            extreme_weather_resilience = self._calculate_extreme_weather_resilience(df)
            
            # Adaptation capacity (proxy based on location characteristics)
            adaptation_capacity = self._estimate_adaptation_capacity(district_data)
            
            # Calculate CRS score
            crs_score = (
                temperature_resilience * 0.30 +
                precipitation_resilience * 0.30 +
                extreme_weather_resilience * 0.25 +
                adaptation_capacity * 0.15
            )
            
            crs_class = self._classify_crs_score(crs_score)
            
            return ClimateResilienceScore(
                temperature_resilience=temperature_resilience,
                precipitation_resilience=precipitation_resilience,
                extreme_weather_resilience=extreme_weather_resilience,
                adaptation_capacity=adaptation_capacity,
                crs_score=round(crs_score, 2),
                crs_class=crs_class
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating CRS: {str(e)}")
            # Return default CRS on error
            return ClimateResilienceScore(
                temperature_resilience=70.0,
                precipitation_resilience=70.0,
                extreme_weather_resilience=65.0,
                adaptation_capacity=70.0,
                crs_score=68.8,
                crs_class=CRSClass.RESILIEN
            )
    
    def _calculate_extreme_weather_resilience(self, df: pd.DataFrame) -> float:
        """Calculate resilience to extreme weather events"""
        try:
            resilience_scores = []
            
            # Temperature extreme resilience
            if 'T2M' in df.columns:
                temp_mean = df['T2M'].mean()
                temp_std = df['T2M'].std()
                temp_extremes = len(df[abs(df['T2M'] - temp_mean) > 2.5 * temp_std])
                temp_resilience = max(40, 100 - (temp_extremes / len(df)) * 200)
                resilience_scores.append(temp_resilience)
            
            # Precipitation extreme resilience
            if 'PRECTOTCORR' in df.columns:
                precip_95th = df['PRECTOTCORR'].quantile(0.95)
                precip_5th = df['PRECTOTCORR'].quantile(0.05)
                
                extreme_wet = len(df[df['PRECTOTCORR'] > precip_95th])
                extreme_dry = len(df[df['PRECTOTCORR'] < precip_5th])
                
                precip_resilience = max(40, 100 - ((extreme_wet + extreme_dry) / len(df)) * 100)
                resilience_scores.append(precip_resilience)
            
            if resilience_scores:
                return np.mean(resilience_scores)
            else:
                return 65.0
                
        except Exception as e:
            self.logger.error(f"Error calculating extreme weather resilience: {str(e)}")
            return 65.0
    
    def _estimate_adaptation_capacity(self, district_data: Dict[str, Any]) -> float:
        """Estimate adaptation capacity based on district characteristics"""
        try:
            # Base adaptation capacity
            base_capacity = 70.0
            
            # Check district characteristics
            district_name = district_data.get('NAME_3', district_data.get('NAME_2', ''))
            
            # Urban areas might have better adaptation infrastructure
            if any(keyword in district_name.lower() for keyword in ['kota', 'banda', 'lhokseumawe']):
                base_capacity += 10
            
            # Coastal areas might have different adaptation challenges
            if any(keyword in district_name.lower() for keyword in ['pantai', 'pesisir', 'pulau']):
                base_capacity -= 5  # Slightly lower due to sea level rise risks
            
            # Agricultural regions might have developed specific adaptations
            if any(keyword in district_name.lower() for keyword in ['utara', 'besar', 'pidie']):
                base_capacity += 5  # Known agricultural areas
            
            return min(100, max(50, base_capacity))
            
        except Exception as e:
            self.logger.error(f"Error estimating adaptation capacity: {str(e)}")
            return 70.0
    
    def _calculate_fsci_score(self, 
                            pci: ProductionCapacityIndex, 
                            psi: ProductionStabilityIndex, 
                            crs: ClimateResilienceScore) -> float:
        """Calculate Food Security Composite Index score"""
        try:
            fsci = (
                pci.pci_score * self.fsci_weights['pci'] +
                psi.psi_score * self.fsci_weights['psi'] +
                crs.crs_score * self.fsci_weights['crs']
            )
            
            return round(fsci, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating FSCI score: {str(e)}")
            return 65.0
    
    def _classify_pci_score(self, score: float) -> PCIClass:
        """Classify PCI score"""
        if score >= 85:
            return PCIClass.SANGAT_TINGGI
        elif score >= 70:
            return PCIClass.TINGGI
        elif score >= 55:
            return PCIClass.SEDANG
        elif score >= 40:
            return PCIClass.RENDAH
        else:
            return PCIClass.SANGAT_RENDAH
    
    def _classify_psi_score(self, score: float) -> PSIClass:
        """Classify PSI score"""
        if score >= 80:
            return PSIClass.SANGAT_STABIL
        elif score >= 65:
            return PSIClass.STABIL
        elif score >= 50:
            return PSIClass.KURANG_STABIL
        else:
            return PSIClass.TIDAK_STABIL
    
    def _classify_crs_score(self, score: float) -> CRSClass:
        """Classify CRS score"""
        if score >= 75:
            return CRSClass.SANGAT_RESILIEN
        elif score >= 60:
            return CRSClass.RESILIEN
        elif score >= 45:
            return CRSClass.KURANG_RESILIEN
        else:
            return CRSClass.TIDAK_RESILIEN
    
    def _classify_fsci_score(self, score: float) -> FSCIClass:
        """Classify FSCI score"""
        if score >= 80:
            return FSCIClass.LUMBUNG_PANGAN_PRIMER
        elif score >= 60:
            return FSCIClass.LUMBUNG_PANGAN_SEKUNDER
        elif score >= 40:
            return FSCIClass.ZONA_POTENSIAL
        else:
            return FSCIClass.BUKAN_PRIORITAS
    
    def _generate_investment_recommendation(self, 
                                          pci: ProductionCapacityIndex,
                                          psi: ProductionStabilityIndex, 
                                          crs: ClimateResilienceScore,
                                          fsci_score: float) -> str:
        """Generate investment recommendation based on analysis"""
        try:
            recommendations = []
            
            # PCI-based recommendations
            if pci.pci_score >= 80:
                recommendations.append("Prioritas pengembangan infrastruktur produksi")
            elif pci.pci_score >= 60:
                recommendations.append("Tingkatkan teknologi pertanian dan irigasi")
            else:
                recommendations.append("Evaluasi kesesuaian lahan dan perbaikan dasar")
            
            # PSI-based recommendations  
            if psi.psi_score < 65:
                recommendations.append("Implementasi sistem mitigasi risiko iklim")
            
            # CRS-based recommendations
            if crs.crs_score < 60:
                recommendations.append("Investasi infrastruktur adaptasi perubahan iklim")
            
            # Overall FSCI recommendations
            if fsci_score >= 80:
                recommendations.append("Kembangkan sebagai lumbung pangan regional")
            elif fsci_score >= 60:
                recommendations.append("Perkuat dukungan untuk mencapai status primer")
            else:
                recommendations.append("Program intensif peningkatan ketahanan pangan")
            
            return " | ".join(recommendations) if recommendations else "Evaluasi lebih lanjut diperlukan"
            
        except Exception as e:
            self.logger.error(f"Error generating investment recommendation: {str(e)}")
            return "Evaluasi lebih lanjut diperlukan"