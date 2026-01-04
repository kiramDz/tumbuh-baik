import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class FSIClass(Enum):
    SANGAT_TINGGI = "Sangat Tinggi"
    TINGGI = "Tinggi"
    SEDANG = "Sedang"
    RENDAH = "Rendah"
    SANGAT_RENDAH = "Sangat Rendah"
    
@dataclass
class FoodSecurityAnalysis:
    """Food Security Index analysis"""
    district_name: str
    district_code: str
    fsi_score: float
    fsi_class: FSIClass
    natural_resources_score: float
    availability_score: float
    analysis_timestamp: str
    
class FoodSecurityAnalyzer:
    """
    Food Security Index Analyzer with Hybrid Classification System
    
    Component 1: Natural Resources & Resilience (60%)
    Component 2: Availability (40%)
    
    Classification: Hybrid system (Percentile + BPS validation)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.fsi_weights = {
            'natural_resources': 0.60,
            'availability': 0.40
        }
        
        # BPS Production Data (2018-2024 average) - Single Source of Truth
        self.bps_production_data = {
            'Aceh Utara': 346449.74,
            'Pidie': 230134.03,
            'Aceh Besar': 190378.44,
            'Bireuen': 157705.34,
            'Aceh Jaya': 52403.47
        }
        
        self.kecamatan_to_kabupaten = {
            'Lhoksukon': 'Aceh Utara',
            'Juli': 'Bireuen', 
            'KotaJuang': 'Bireuen',
            'Indrapuri': 'Aceh Besar',
            'Montasik': 'Aceh Besar', 
            'Darussalam': 'Aceh Besar',
            'Jaya': 'Pidie',
            'Pidie': 'Pidie',
            'Indrajaya': 'Pidie',
            'Teunom': 'Aceh Jaya',
            'SetiaBakti': 'Aceh Jaya'
        }
        
        self.logger.info("Food Security Index analyzer initialized (FSI with Hybrid Classification)")
        
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
            FoodSecurityAnalysis object with initial classification
            (Final classification will be applied by hybrid system)
        """
        try:
            district_name = district_data.get('NAME_3', district_data.get('NAME_2', 'Unknown District'))
            district_code = district_data.get('GID_3', district_data.get('GID_2', 'Unknown'))
            
            self.logger.info(f"Starting FSI analysis for {district_name}")
            
            # 1. Natural Resources & Resilience Score
            natural_resources_score = self._calculate_natural_resources_score(
                climate_time_series, base_suitability_score
            )
            
            # 2. Availability Score
            availability_score = self._calculate_availability_score(
                climate_time_series, base_suitability_score
            )
            
            # 3. Food Security Index
            fsi_score = self._calculate_fsi_score(natural_resources_score, availability_score)
            
            # 4. Initial classification (will be overridden by hybrid system)
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
                           f"FSI={fsi_score:.1f} (initial: {fsi_class.value})")
            
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
            
            # Climate stability (40% weight)
            stability_score = 75.0
            if 'T2M' in df.columns and 'PRECTOTCORR' in df.columns:
                temp_cv = (df['T2M'].std() / df['T2M'].mean()) * 100
                precip_cv = (df['PRECTOTCORR'].std() / df['PRECTOTCORR'].mean()) * 100 if df['PRECTOTCORR'].mean() > 0 else 30
                
                temp_stability = max(50, 100 - (temp_cv * 10))
                precip_stability = max(50, 100 - (precip_cv * 2))
                
                stability_score = (temp_stability + precip_stability) / 2
            
            natural_resources = (climate_sustainability * 0.6) + (stability_score * 0.4)
            
            return min(100, max(0, natural_resources))
        except Exception as e:
            self.logger.error(f"Error calculating natural resources score: {str(e)}")
            return base_suitability * 0.8
        
    def _calculate_availability_score(self,
                                    climate_data: List[Dict[str, Any]],
                                    base_suitability: float) -> float:
        """Calculate Availability score"""
        try:
            if not climate_data:
                return base_suitability * 0.85
            
            df = pd.DataFrame(climate_data)
            
            # Water availability
            water_score = 70.0
            if 'PRECTOTCORR' in df.columns:
                avg_precip = df['PRECTOTCORR'].mean()
                annual_precip = avg_precip * 365
                
                if 1200 <= annual_precip <= 1800:
                    water_score = 100
                elif annual_precip > 1800:
                    water_score = max(60, 100 - ((annual_precip - 1800) / 1000) * 30)
                else:
                    water_score = max(40, (annual_precip / 1200) * 100)
            
            # Temperature suitability
            temp_score = 75.0
            if 'T2M' in df.columns:
                avg_temp = df['T2M'].mean()
                
                if 24 <= avg_temp <= 30:
                    temp_score = 100
                elif avg_temp > 30:
                    temp_score = max(60, 100 - ((avg_temp - 30) / 5) * 40)
                else:
                    temp_score = max(50, ((avg_temp - 20) / 4) * 100)
            
            availability = (water_score * 0.6) + (temp_score * 0.4)
            
            return min(100, max(30, availability))
            
        except Exception as e:
            self.logger.error(f"Error calculating availability score: {str(e)}")
            return base_suitability * 0.75
        
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
        """
        Initial FSI classification (relaxed thresholds for logging purposes)
        
        NOTE: This is TEMPORARY classification for debugging.
        Final classification is determined by hybrid system:
        - apply_percentile_based_classification() for climate-based
        - apply_bps_calibrated_classification() for production-based
        """
        # Relaxed thresholds based on actual data distribution (62-73)
        if score >= 70:
            return FSIClass.SANGAT_TINGGI
        elif score >= 67:
            return FSIClass.TINGGI
        elif score >= 65:
            return FSIClass.SEDANG
        elif score >= 63:
            return FSIClass.RENDAH
        else:
            return FSIClass.SANGAT_RENDAH
                
                
    def apply_percentile_based_classification(self, fsi_results: List[FoodSecurityAnalysis]) -> List[FoodSecurityAnalysis]:
        """Apply percentile-based FSI classification - 18-18-27-18-18 distribution"""
        try:
            if not fsi_results or len(fsi_results) == 0:
                return fsi_results
                
            # Extract FSI scores and create ranked list
            scored_results = [(result, result.fsi_score) for result in fsi_results]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            total_regions = len(scored_results)
            
            if total_regions < 5:
                self.logger.warning("Too few regions for percentile classification")
                return fsi_results
            
            # Calculate split indices for 18-18-27-18-18 distribution
            split_1 = int(np.ceil(total_regions * 0.18))
            split_2 = int(np.ceil(total_regions * 0.36))
            split_3 = int(np.ceil(total_regions * 0.63))
            split_4 = int(np.ceil(total_regions * 0.81))
            
            self.logger.info(f"Percentile splits (18-18-27-18-18): "
                            f"ST=[0-{split_1-1}], T=[{split_1}-{split_2-1}], "
                            f"S=[{split_2}-{split_3-1}], R=[{split_3}-{split_4-1}], "
                            f"SR=[{split_4}-{total_regions-1}]")
            
            # Apply classification based on position
            for idx, (result, score) in enumerate(scored_results):
                if idx < split_1:
                    result.fsi_class = FSIClass.SANGAT_TINGGI
                    class_name = "SANGAT TINGGI"
                elif idx < split_2:
                    result.fsi_class = FSIClass.TINGGI
                    class_name = "TINGGI"
                elif idx < split_3:
                    result.fsi_class = FSIClass.SEDANG
                    class_name = "SEDANG"
                elif idx < split_4:
                    result.fsi_class = FSIClass.RENDAH
                    class_name = "RENDAH"
                else:
                    result.fsi_class = FSIClass.SANGAT_RENDAH
                    class_name = "SANGAT RENDAH"
                
                self.logger.info(f"  [{idx}] {result.district_name}: {score:.1f} → {class_name}")
            
            # Log final distribution
            class_counts = {}
            for result in fsi_results:
                class_name = result.fsi_class.value
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            self.logger.info(f"Percentile FSI distribution: {class_counts}")
            
            return fsi_results
            
        except Exception as e:
            self.logger.error(f"Error in percentile classification: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return fsi_results
    
    
    def apply_bps_calibrated_classification(self, fsi_results: List[FoodSecurityAnalysis]) -> List[FoodSecurityAnalysis]:
        """Apply BPS production-calibrated FSI classification"""
        try:
            # Create production-based ranking
            kabupaten_ranked = sorted(self.bps_production_data.items(), key=lambda x: x[1], reverse=True)
            production_ranking = {kabupaten: rank for rank, (kabupaten, _) in enumerate(kabupaten_ranked, 1)}
            
            self.logger.info("BPS Production Ranking:")
            for kabupaten, rank in sorted(production_ranking.items(), key=lambda x: x[1]):
                production = self.bps_production_data[kabupaten]
                self.logger.info(f"  Rank {rank}: {kabupaten} - {production:.0f} tons/year")
            
            # Apply BPS-calibrated classification
            for result in fsi_results:
                kecamatan = result.district_name
                kabupaten = self.kecamatan_to_kabupaten.get(kecamatan)
                
                if kabupaten and kabupaten in production_ranking:
                    production_rank = production_ranking[kabupaten]
                    production_tons = self.bps_production_data[kabupaten]
                    
                    # Classification based on production ranking
                    if production_rank == 1:
                        result.fsi_class = FSIClass.SANGAT_TINGGI
                        calibrated_class = "SANGAT TINGGI"
                    elif production_rank == 2:
                        result.fsi_class = FSIClass.TINGGI
                        calibrated_class = "TINGGI"
                    elif production_rank == 3:
                        result.fsi_class = FSIClass.SEDANG
                        calibrated_class = "SEDANG"
                    elif production_rank == 4:
                        result.fsi_class = FSIClass.RENDAH
                        calibrated_class = "RENDAH"
                    else:
                        result.fsi_class = FSIClass.SANGAT_RENDAH
                        calibrated_class = "SANGAT RENDAH"
                    
                    self.logger.info(f"  BPS: {kecamatan} ({kabupaten}: {production_tons:.0f}t, rank {production_rank}) → {calibrated_class}")
            
            # Log final distribution
            class_counts = {}
            for result in fsi_results:
                class_name = result.fsi_class.value
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            self.logger.info(f"BPS-Calibrated FSI distribution: {class_counts}")
            
            return fsi_results
            
        except Exception as e:
            self.logger.error(f"Error in BPS calibration: {str(e)}")
            return fsi_results

    def apply_hybrid_fsi_classification(self, fsi_results: List[FoodSecurityAnalysis]) -> List[FoodSecurityAnalysis]:
        """Apply hybrid FSI classification: percentile + BPS validation"""
        try:
            self.logger.info("="*70)
            self.logger.info("HYBRID FSI CLASSIFICATION SYSTEM")
            self.logger.info("="*70)
            
            # Step 1: Apply percentile classification
            self.logger.info("Step 1: Applying percentile-based classification (climate-based)")
            fsi_results = self.apply_percentile_based_classification(fsi_results)
            
            # Step 2: Calculate correlation
            self.logger.info("\nStep 2: Calculating FSI-Production correlation")
            correlation = self._calculate_fsi_production_correlation(fsi_results)
            
            # Step 3: Decision
            self.logger.info(f"\nStep 3: Correlation assessment: {correlation:.3f}")
            
            if correlation < 0.6:
                self.logger.warning(f"⚠️  Weak correlation ({correlation:.3f} < 0.6)")
                self.logger.warning("→ Switching to BPS-calibrated classification (production-based)\n")
                fsi_results = self.apply_bps_calibrated_classification(fsi_results)
            else:
                self.logger.info(f"✅ Good correlation ({correlation:.3f} ≥ 0.6)")
                self.logger.info("→ Keeping percentile classification (climate-based)")
            
            self.logger.info("="*70)
            
            return fsi_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid classification: {str(e)}")
            return fsi_results

    def _calculate_fsi_production_correlation(self, fsi_results: List[FoodSecurityAnalysis]) -> float:
        """
        Calculate correlation between FSI and BPS production at KABUPATEN level
        
        Fixed: Aggregates FSI scores by kabupaten before correlation calculation
        to avoid duplicate production values inflating correlation.
        """
        try:
            import scipy.stats as stats
            
            # Aggregate FSI by kabupaten
            kabupaten_fsi = {}
            kabupaten_counts = {}
            
            for result in fsi_results:
                kabupaten = self.kecamatan_to_kabupaten.get(result.district_name)
                if kabupaten:
                    if kabupaten not in kabupaten_fsi:
                        kabupaten_fsi[kabupaten] = 0
                        kabupaten_counts[kabupaten] = 0
                    kabupaten_fsi[kabupaten] += result.fsi_score
                    kabupaten_counts[kabupaten] += 1
            
            # Calculate average FSI per kabupaten
            fsi_scores = []
            production_values = []
            
            for kabupaten in self.bps_production_data.keys():
                if kabupaten in kabupaten_fsi:
                    avg_fsi = kabupaten_fsi[kabupaten] / kabupaten_counts[kabupaten]
                    fsi_scores.append(avg_fsi)
                    production_values.append(self.bps_production_data[kabupaten])
                    self.logger.info(f"  {kabupaten}: FSI={avg_fsi:.1f}, Production={self.bps_production_data[kabupaten]:.0f}t")
            
            if len(fsi_scores) >= 3:
                correlation, p_value = stats.spearmanr(fsi_scores, production_values)
                self.logger.info(f"  Spearman ρ = {correlation:.3f} (p={p_value:.3f})")
                return correlation
            else:
                self.logger.warning("Insufficient kabupaten for correlation")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
        
    def validate_fsi_with_production(self, fsi_results: List[FoodSecurityAnalysis], 
                                    bps_production_data: Dict[str, Any]) -> float:
        """Legacy method - kept for backward compatibility"""
        return self._calculate_fsi_production_correlation(fsi_results)