import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from services.food_security_analyzer import (
    FoodSecurityAnalyzer, FoodSecurityAnalysis, FSIClass
)
from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
from services.bps_api_service import BPSApiService, RiceProductionData

logger = logging.getLogger(__name__)

@dataclass
class KecamatanAnalysis:
    """Kecamatan-level food security analysis (Level 1) - FSI Only"""
    nasa_location_name: str
    kecamatan_name: str
    kabupaten_name: str
    area_km2: float
    area_weight: float
    fsi_analysis: FoodSecurityAnalysis 
    analysis_level: str = "kecamatan"
    
@dataclass
class KabupatenValidation:
    """Kabupaten-level BPS validation data"""
    kabupaten_name: str
    bps_name: str
    bps_historical_data: Dict[int, RiceProductionData]
    latest_production_tons: float
    average_production_tons: float
    production_trend: str
    data_years_available: List[int]
    data_coverage_years: int
    

@dataclass 
class KabupatenAnalysis:
    """Kabupaten-level aggregated analysis (Level 2) - FSI Only"""
    kabupaten_name: str
    bps_compatible_name: str
    total_area_km2: float
    sample_kecamatan: List[str]
    constituent_nasa_locations: List[str]
    
    # Simplified FSI metrics only
    aggregated_fsi_score: float           # ← Updated: single FSI score
    aggregated_fsi_class: FSIClass        # ← Updated: single FSI class
    natural_resources_score: float        # ← Updated: component 1
    availability_score: float             # ← Updated: component 2
    
    # Component analyses
    kecamatan_analyses: List[KecamatanAnalysis]
    bps_validation: KabupatenValidation
    
    # Simplified validation metrics
    climate_production_correlation: float
    production_efficiency_score: float
    climate_potential_rank: int
    actual_production_rank: int
    performance_gap_category: str
    
    validation_notes: str
    analysis_level: str = "kabupaten"
    
@dataclass
class TwoLevelAnalysisResult:
    """Complete two-level analysis result - Simplified"""
    analysis_timestamp: str
    analysis_period: str
    bps_data_period: str
    
    # Level summaries
    level_1_kecamatan_count: int
    level_2_kabupaten_count: int
    
    # Results
    kecamatan_analyses: List[KecamatanAnalysis]
    kabupaten_analyses: List[KabupatenAnalysis]
    
    # Simplified insights
    cross_level_insights: Dict[str, Any]
    
    # Rankings
    kabupaten_climate_ranking: List[Dict[str, Any]]
    kabupaten_production_ranking: List[Dict[str, Any]]
    
    methodology_summary: str


class TwoLevelFoodSecurityAnalyzer:
    """
    Simplified Two-Level Food Security Analyzer - FSI Only
    Level 1: Kecamatan climate-based FSI analysis (NASA POWER locations)
    Level 2: Kabupaten BPS-validated FSI analysis (Real production data)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize component services
        self.kecamatan_analyzer = FoodSecurityAnalyzer()
        self.mapping_service = KecamatanKabupatenMappingService()
        self.bps_service = BPSApiService()
        
        # Aggregation configuration
        self.aggregation_method = "area_weighted_average"
        
        self.logger.info("Two-Level Food Security Analyzer initialized (FSI Only)")
    
    def perform_two_level_analysis(self, 
                                  spatial_analysis_results: List[Dict[str, Any]],
                                  bps_start_year: int = 2018,
                                  bps_end_year: int = 2024) -> TwoLevelAnalysisResult:
        """
        Perform complete two-level food security analysis - FSI Only
        
        Args:
            spatial_analysis_results: Results from spatial analysis (NASA location level)
            bps_start_year: Start year for BPS data
            bps_end_year: End year for BPS data
            
        Returns:
            Complete TwoLevelAnalysisResult
        """
        try:
            self.logger.info("Starting two-level FSI analysis")
            self.logger.info(f"Input: {len(spatial_analysis_results)} NASA location results")
            self.logger.info(f"BPS period: {bps_start_year}-{bps_end_year}")
            
            # Level 1: Process kecamatan analyses from NASA locations
            kecamatan_analyses = self._process_level_1_kecamatan_analyses(spatial_analysis_results)
            self.logger.info(f"Level 1 complete: {len(kecamatan_analyses)} kecamatan analyses")
            
            # Level 2: Aggregate to kabupaten and validate with BPS
            kabupaten_analyses = self._process_level_2_kabupaten_analyses(
                kecamatan_analyses, bps_start_year, bps_end_year
            )
            self.logger.info(f"Level 2 complete: {len(kabupaten_analyses)} kabupaten analyses")
            
            # Generate cross-level insights
            cross_level_insights = self._generate_cross_level_insights(
                kecamatan_analyses, kabupaten_analyses
            )
            
            # Generate rankings
            climate_ranking, production_ranking = self._generate_rankings(kabupaten_analyses)
            
            # Create final result
            result = TwoLevelAnalysisResult(
                analysis_timestamp=datetime.now().isoformat(),
                analysis_period=f"NASA Climate Analysis: Current",
                bps_data_period=f"BPS Production Data: {bps_start_year}-{bps_end_year}",
                level_1_kecamatan_count=len(kecamatan_analyses),
                level_2_kabupaten_count=len(kabupaten_analyses),
                kecamatan_analyses=kecamatan_analyses,
                kabupaten_analyses=kabupaten_analyses,
                cross_level_insights=cross_level_insights,
                kabupaten_climate_ranking=climate_ranking,
                kabupaten_production_ranking=production_ranking,
                methodology_summary=self._generate_methodology_summary()
            )
            
            self.logger.info(f"Two-level FSI analysis complete: {len(kecamatan_analyses)} kecamatan → {len(kabupaten_analyses)} kabupaten")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in two-level FSI analysis: {str(e)}")
            raise
    
    def _process_level_1_kecamatan_analyses(self, 
                                          spatial_results: List[Dict[str, Any]]) -> List[KecamatanAnalysis]:
        """Process Level 1: Kecamatan climate-based FSI analyses from NASA locations"""
        try:
            kecamatan_analyses = []
            
            for result in spatial_results:
                nasa_location_name = result.get('district')  # Your spatial results use 'district' field
                
                # Get kecamatan mapping info
                kecamatan_info = self.mapping_service.get_kecamatan_info(nasa_location_name)
                
                if not kecamatan_info:
                    self.logger.warning(f"No mapping found for NASA location: {nasa_location_name}")
                    continue
                
                # Convert spatial result to FoodSecurityAnalysis
                fsi_analysis = self._convert_spatial_to_fsi_analysis(result)
                
                # Create KecamatanAnalysis with real mapping data
                kecamatan_analysis = KecamatanAnalysis(
                    nasa_location_name=nasa_location_name,
                    kecamatan_name=kecamatan_info.kecamatan_name,
                    kabupaten_name=kecamatan_info.kabupaten_name,
                    area_km2=kecamatan_info.area_km2,
                    area_weight=kecamatan_info.area_weight,
                    fsi_analysis=fsi_analysis  # ← Updated: renamed from fsci_analysis
                )
                
                kecamatan_analyses.append(kecamatan_analysis)
                
                self.logger.info(f"Level 1: {nasa_location_name} → {kecamatan_info.kecamatan_name} "
                               f"(FSI: {fsi_analysis.fsi_score}) [{kecamatan_info.kabupaten_name}]")
            
            return kecamatan_analyses
            
        except Exception as e:
            self.logger.error(f"Error in Level 1 processing: {str(e)}")
            raise
    
    def _process_level_2_kabupaten_analyses(self, 
                                          kecamatan_analyses: List[KecamatanAnalysis],
                                          bps_start_year: int,
                                          bps_end_year: int) -> List[KabupatenAnalysis]:
        """Process Level 2: Kabupaten aggregation and BPS validation"""
        try:
            # Group kecamatan by kabupaten
            kabupaten_groups = {}
            for analysis in kecamatan_analyses:
                kabupaten = analysis.kabupaten_name
                if kabupaten not in kabupaten_groups:
                    kabupaten_groups[kabupaten] = []
                kabupaten_groups[kabupaten].append(analysis)
            
            kabupaten_analyses = []
            
            for kabupaten_name, kecamatan_group in kabupaten_groups.items():
                try:
                    self.logger.info(f"Processing Level 2 for {kabupaten_name}: {len(kecamatan_group)} kecamatan")
                    
                    # Get kabupaten info
                    kabupaten_info = self.mapping_service.get_kabupaten_info(kabupaten_name)
                    if not kabupaten_info:
                        self.logger.error(f"No kabupaten info found for {kabupaten_name}")
                        continue
                    
                    # Aggregate FSI scores using real area weights
                    aggregated_metrics = self._aggregate_kecamatan_metrics(kabupaten_name, kecamatan_group)
                    
                    # Get BPS validation data
                    bps_validation = self._fetch_bps_validation_data(
                        kabupaten_info.bps_compatible_name, bps_start_year, bps_end_year
                    )
                    
                    # Calculate climate-production correlation
                    correlation = self._calculate_climate_production_correlation(
                        aggregated_metrics, bps_validation
                    )
                    
                    # Calculate production efficiency
                    efficiency_score = self._calculate_production_efficiency_score(
                        aggregated_metrics, bps_validation
                    )
                    
                    # Create kabupaten analysis
                    kabupaten_analysis = KabupatenAnalysis(
                        kabupaten_name=kabupaten_name,
                        bps_compatible_name=kabupaten_info.bps_compatible_name,
                        total_area_km2=kabupaten_info.total_area_km2,
                        sample_kecamatan=[k.kecamatan_name for k in kecamatan_group],
                        constituent_nasa_locations=[k.nasa_location_name for k in kecamatan_group],
                        
                        # Updated FSI metrics only
                        aggregated_fsi_score=aggregated_metrics['fsi'],
                        aggregated_fsi_class=self.kecamatan_analyzer._classify_fsi_score(aggregated_metrics['fsi']),
                        natural_resources_score=aggregated_metrics['natural_resources'],
                        availability_score=aggregated_metrics['availability'],
                        
                        kecamatan_analyses=kecamatan_group,
                        bps_validation=bps_validation,
                        
                        climate_production_correlation=correlation,
                        production_efficiency_score=efficiency_score,
                        climate_potential_rank=0,  # Will be set in ranking phase
                        actual_production_rank=0,  # Will be set in ranking phase
                        performance_gap_category=self._determine_performance_gap_category(
                            aggregated_metrics['fsi'], efficiency_score
                        ),
                        
                        validation_notes=self._generate_validation_notes(
                            aggregated_metrics, bps_validation, correlation
                        )
                    )
                    
                    kabupaten_analyses.append(kabupaten_analysis)
                    
                    self.logger.info(f"Level 2 complete for {kabupaten_name}: "
                                   f"FSI={aggregated_metrics['fsi']:.1f}, "
                                   f"Production={bps_validation.latest_production_tons:.0f}t, "
                                   f"Correlation={correlation:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing kabupaten {kabupaten_name}: {str(e)}")
                    continue
            
            return kabupaten_analyses
            
        except Exception as e:
            self.logger.error(f"Error in Level 2 processing: {str(e)}")
            raise
    
    def _aggregate_kecamatan_metrics(self, 
                                   kabupaten_name: str, 
                                   kecamatan_analyses: List[KecamatanAnalysis]) -> Dict[str, float]:
        """Aggregate kecamatan FSI scores to kabupaten level using real area weights"""
        try:
            # Use area-weighted aggregation for FSI only
            weighted_fsi = 0.0
            weighted_natural_resources = 0.0
            weighted_availability = 0.0
            total_weight = 0.0
            
            for analysis in kecamatan_analyses:
                weight = analysis.area_weight
                
                # Updated to use FSI analysis structure
                weighted_fsi += analysis.fsi_analysis.fsi_score * weight
                weighted_natural_resources += analysis.fsi_analysis.natural_resources_score * weight
                weighted_availability += analysis.fsi_analysis.availability_score * weight
                total_weight += weight
            
            if total_weight == 0:
                self.logger.warning(f"Zero total weight for {kabupaten_name}")
                return {
                    'fsi': 65.0, 
                    'natural_resources': 70.0, 
                    'availability': 60.0
                }
            
            aggregated_metrics = {
                'fsi': round(weighted_fsi / total_weight, 2),
                'natural_resources': round(weighted_natural_resources / total_weight, 2),
                'availability': round(weighted_availability / total_weight, 2)
            }
            
            self.logger.info(f"Aggregated FSI metrics for {kabupaten_name}: "
                           f"FSI={aggregated_metrics['fsi']}, weights_sum={total_weight:.3f}")
            
            return aggregated_metrics
                
        except Exception as e:
            self.logger.error(f"Error aggregating FSI metrics for {kabupaten_name}: {str(e)}")
            return {
                'fsi': 65.0, 
                'natural_resources': 70.0, 
                'availability': 60.0
            }
    
    def _fetch_bps_validation_data(self, 
                                  bps_kabupaten_name: str,
                                  start_year: int,
                                  end_year: int) -> KabupatenValidation:
        """Fetch BPS production data for kabupaten validation"""
        try:
            self.logger.info(f"Fetching BPS data for {bps_kabupaten_name} ({start_year}-{end_year})")
            
            # Fetch historical BPS data
            historical_data = self.bps_service.fetch_kabupaten_historical_data(
                bps_kabupaten_name, start_year, end_year
            )
            
            if not historical_data:
                self.logger.warning(f"No BPS data found for {bps_kabupaten_name}")
                return self._create_default_bps_validation(bps_kabupaten_name)
            
            # Calculate metrics
            productions = [record.padi_ton for record in historical_data.values()]  # ← Updated: use padi_ton
            latest_production = productions[-1] if productions else 0.0
            average_production = np.mean(productions) if productions else 0.0
            
            # Determine trend
            trend = self._calculate_production_trend(productions)
            
            validation = KabupatenValidation(
                kabupaten_name=bps_kabupaten_name,
                bps_name=bps_kabupaten_name,
                bps_historical_data=historical_data,
                latest_production_tons=latest_production,
                average_production_tons=average_production,
                production_trend=trend,
                data_years_available=sorted(historical_data.keys()),
                data_coverage_years=len(historical_data)
            )
            
            self.logger.info(f"BPS data for {bps_kabupaten_name}: "
                           f"{len(historical_data)} years, latest: {latest_production:.0f}t")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error fetching BPS data for {bps_kabupaten_name}: {str(e)}")
            return self._create_default_bps_validation(bps_kabupaten_name)
    
    def _calculate_production_trend(self, productions: List[float]) -> str:
        """Calculate production trend from historical data"""
        if len(productions) < 2:
            return "insufficient_data"
        
        # Linear regression slope
        years = list(range(len(productions)))
        slope = np.polyfit(years, productions, 1)[0]
        
        if slope > 2000:  # Increase > 2000 tons per year
            return "increasing"
        elif slope < -2000:  # Decrease > 2000 tons per year
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_climate_production_correlation(self, 
                                                aggregated_metrics: Dict[str, float],
                                                bps_validation: KabupatenValidation) -> float:
        """Calculate correlation between climate potential and actual production"""
        try:
            # Normalize climate score (FSI) to production scale
            fsi_score = aggregated_metrics['fsi']  # ← Updated: use FSI score
            
            # Normalize production to expected range (based on Aceh rice production knowledge)
            # Max expected production per kabupaten ~ 400,000 tons
            max_expected = 400000
            normalized_production = min(100, (bps_validation.latest_production_tons / max_expected) * 100)
            
            # Simple correlation calculation (can be enhanced)
            score_diff = abs(fsi_score - normalized_production)
            correlation = max(0, (100 - score_diff) / 100)
            
            return round(correlation, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return 0.500
    
    def _calculate_production_efficiency_score(self, 
                                             aggregated_metrics: Dict[str, float],
                                             bps_validation: KabupatenValidation) -> float:
        """Calculate production efficiency relative to climate potential"""
        try:
            fsi_score = aggregated_metrics['fsi']  # ← Updated: use FSI score
            actual_production = bps_validation.latest_production_tons
            
            # Expected production based on FSI (rough regional benchmark)
            # FSI 80 = 350,000 tons, FSI 60 = 200,000 tons (linear scaling)
            expected_production = ((fsi_score - 50) / 30) * 200000 + 150000
            expected_production = max(100000, expected_production)  # Minimum threshold
            
            # Efficiency calculation
            efficiency = (actual_production / expected_production) * 100
            
            return round(min(200, max(20, efficiency)), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency: {str(e)}")
            return 75.0
    
    def _determine_performance_gap_category(self, fsi_score: float, efficiency_score: float) -> str:
        """Determine performance gap category"""
        gap = efficiency_score - fsi_score
        
        if gap > 15:
            return "overperforming"
        elif gap < -15:
            return "underperforming" 
        else:
            return "aligned"
    
    def _generate_rankings(self, kabupaten_analyses: List[KabupatenAnalysis]) -> tuple:
        """Generate climate and production rankings"""
        # Climate potential ranking (FSI)
        climate_sorted = sorted(kabupaten_analyses, key=lambda x: x.aggregated_fsi_score, reverse=True)
        climate_ranking = []
        
        for rank, analysis in enumerate(climate_sorted, 1):
            analysis.climate_potential_rank = rank  # Update rank in object
            climate_ranking.append({
                "rank": rank,
                "kabupaten_name": analysis.kabupaten_name,
                "fsi_score": analysis.aggregated_fsi_score,        # ← Updated: FSI score
                "fsi_class": analysis.aggregated_fsi_class.value,  # ← Updated: FSI class
                "sample_kecamatan": len(analysis.sample_kecamatan)
            })
        
        # Production ranking
        production_sorted = sorted(kabupaten_analyses, 
                                 key=lambda x: x.bps_validation.latest_production_tons, reverse=True)
        production_ranking = []
        
        for rank, analysis in enumerate(production_sorted, 1):
            analysis.actual_production_rank = rank  # Update rank in object
            production_ranking.append({
                "rank": rank,
                "kabupaten_name": analysis.kabupaten_name,
                "latest_production_tons": analysis.bps_validation.latest_production_tons,
                "production_trend": analysis.bps_validation.production_trend,
            })
        
        return climate_ranking, production_ranking
    
    # Helper methods...
    def _convert_spatial_to_fsi_analysis(self, spatial_result: Dict[str, Any]) -> FoodSecurityAnalysis:
        """Convert spatial analysis result to FoodSecurityAnalysis object"""
        district = spatial_result.get('district', 'Unknown')
        
        # Extract FSI-related scores from spatial result
        # You may need to adjust these field names based on your actual spatial results
        fsi_score = spatial_result.get('fsi_score', 
                    spatial_result.get('suitability_score', 65.0))  # fallback to suitability_score
        
        natural_resources_score = spatial_result.get('natural_resources_score', fsi_score * 0.9)
        availability_score = spatial_result.get('availability_score', fsi_score * 1.1)
        
        return FoodSecurityAnalysis(
            district_name=district,
            district_code=spatial_result.get('district_code', 'Unknown'),
            fsi_score=fsi_score,
            fsi_class=self.kecamatan_analyzer._classify_fsi_score(fsi_score),
            natural_resources_score=natural_resources_score,
            availability_score=availability_score,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _create_default_bps_validation(self, kabupaten_name: str) -> KabupatenValidation:
        """Create default BPS validation when data is not available"""
        return KabupatenValidation(
            kabupaten_name=kabupaten_name,
            bps_name=kabupaten_name,
            bps_historical_data={},
            latest_production_tons=0.0,
            average_production_tons=0.0,
            production_trend="no_data",
            data_years_available=[],
            data_coverage_years=0
        )
    
    def _generate_validation_notes(self, 
                                 aggregated_metrics: Dict[str, float],
                                 bps_validation: KabupatenValidation,
                                 correlation: float) -> str:
        """Generate validation notes"""
        notes = []
        
        fsi_score = aggregated_metrics['fsi']
        
        # Correlation assessment
        if correlation > 0.7:
            notes.append("Strong climate-production alignment")
        elif correlation > 0.4:
            notes.append("Moderate climate-production correlation")
        else:
            notes.append("Low climate-production correlation")
        
        # Data coverage assessment
        if bps_validation.data_coverage_years >= 6:
            notes.append(f"Good BPS coverage ({bps_validation.data_coverage_years} years)")
        elif bps_validation.data_coverage_years >= 3:
            notes.append(f"Moderate BPS coverage ({bps_validation.data_coverage_years} years)")
        else:
            notes.append("Limited BPS data")
        
        # Production trend
        if bps_validation.production_trend == "increasing":
            notes.append("Production trending up")
        elif bps_validation.production_trend == "decreasing":
            notes.append("Production declining")
        
        return " | ".join(notes)
    
    def _generate_cross_level_insights(self, 
                                     kecamatan_analyses: List[KecamatanAnalysis],
                                     kabupaten_analyses: List[KabupatenAnalysis]) -> Dict[str, Any]:
        """Generate simplified cross-level insights"""
        try:
            insights = {
                "methodology_validation": {},
                "climate_production_alignment": {},
                "spatial_variability": {}
            }
            
            # Methodology validation
            total_correlation = np.mean([k.climate_production_correlation for k in kabupaten_analyses])
            
            insights["methodology_validation"] = {
                "overall_climate_production_correlation": round(total_correlation, 3),
                "validation_strength": "strong" if total_correlation > 0.7 else "moderate" if total_correlation > 0.4 else "weak",
                "sample_size": f"{len(kecamatan_analyses)} kecamatan across {len(kabupaten_analyses)} kabupaten"
            }
            
            # Climate-production alignment analysis
            for analysis in kabupaten_analyses:
                insights["climate_production_alignment"][analysis.kabupaten_name] = {
                    "climate_rank": analysis.climate_potential_rank,
                    "production_rank": analysis.actual_production_rank,
                    "rank_difference": abs(analysis.climate_potential_rank - analysis.actual_production_rank),
                    "performance_category": analysis.performance_gap_category,
                    "correlation": analysis.climate_production_correlation
                }
            
            # Spatial variability within kabupaten
            for analysis in kabupaten_analyses:
                kecamatan_fsi_scores = [k.fsi_analysis.fsi_score for k in analysis.kecamatan_analyses]
                if len(kecamatan_fsi_scores) > 1:
                    fsi_variance = np.var(kecamatan_fsi_scores)
                    
                    insights["spatial_variability"][analysis.kabupaten_name] = {
                        "kecamatan_fsi_range": f"{min(kecamatan_fsi_scores):.1f}-{max(kecamatan_fsi_scores):.1f}",
                        "fsi_variance": round(fsi_variance, 2),
                        "internal_variability": "high" if fsi_variance > 50 else "moderate" if fsi_variance > 15 else "low",
                        "largest_contributor": max(analysis.kecamatan_analyses, key=lambda x: x.area_weight).kecamatan_name
                    }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating cross-level insights: {str(e)}")
            return {}
    
    def _generate_methodology_summary(self) -> str:
        """Generate methodology summary for reporting"""
        return ("Two-level FSI analysis: (1) Kecamatan-level Food Security Index (FSI) analysis "
                "using NASA POWER data at 11 locations mapped to administrative boundaries via GeoJSON, "
                "(2) Area-weighted aggregation to kabupaten level with BPS rice production validation (2018-2024). "
                "FSI components: Natural Resources & Resilience (60%) + Availability (40%). "
                "Climate potential assessed at micro-scale, validated against macro-scale actual production data "
                "using Indonesian administrative hierarchy.")
    