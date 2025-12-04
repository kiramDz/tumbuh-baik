import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from services.food_security_analyzer import (
    FoodSecurityAnalyzer, FoodSecurityAnalysis, 
    FSCIClass, PCIClass, PSIClass, CRSClass
)
from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
from services.bps_api_service import BPSApiService, RiceProductionData

logger = logging.getLogger(__name__)

@dataclass 
class KecamatanAnalysis:
    """Kecamatan-level food security analysis (Level 1)"""
    nasa_location_name: str
    kecamatan_name: str
    kabupaten_name: str
    area_km2: float
    area_weight: float
    fsci_analysis: FoodSecurityAnalysis
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
    """Kabupaten-level aggregated analysis (Level 2)"""
    kabupaten_name: str
    bps_compatible_name: str
    total_area_km2: float
    constituent_kecamatan: List[str]
    constituent_nasa_locations: List[str]
    
    # Aggregated FSCI metrics
    aggregated_fsci_score: float
    aggregated_fsci_class: FSCIClass
    aggregated_pci_score: float
    aggregated_psi_score: float
    aggregated_crs_score: float
    
    # Component analyses
    kecamatan_analyses: List[KecamatanAnalysis]
    bps_validation: KabupatenValidation
    
    # Validation metrics
    climate_production_correlation: float
    production_efficiency_score: float
    climate_potential_rank: int
    actual_production_rank: int
    performance_gap_category: str
    
    validation_notes: str
    analysis_level: str = "kabupaten"

@dataclass
class TwoLevelAnalysisResult:
    """Complete two-level analysis result"""
    analysis_timestamp: str
    analysis_period: str
    bps_data_period: str
    
    # Level summaries
    level_1_kecamatan_count: int
    level_2_kabupaten_count: int
    
    # Results
    kecamatan_analyses: List[KecamatanAnalysis]
    kabupaten_analyses: List[KabupatenAnalysis]
    
    # Cross-level insights
    cross_level_insights: Dict[str, Any]
    
    # Rankings
    kabupaten_climate_ranking: List[Dict[str, Any]]
    kabupaten_production_ranking: List[Dict[str, Any]]
    
    methodology_summary: str

class TwoLevelFoodSecurityAnalyzer:
    """
    Two-Level Hybrid Food Security Analyzer
    Level 1: Kecamatan climate-based analysis (NASA POWER locations)
    Level 2: Kabupaten BPS-validated analysis (Real production data)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize component services
        self.kecamatan_analyzer = FoodSecurityAnalyzer()
        self.mapping_service = KecamatanKabupatenMappingService()
        self.bps_service = BPSApiService()
        
        # Aggregation configuration
        self.aggregation_method = "area_weighted_average"
        
        self.logger.info("Two-Level Food Security Analyzer initialized with real GeoJSON area weights")
    
    def perform_two_level_analysis(self, 
                                  spatial_analysis_results: List[Dict[str, Any]],
                                  bps_start_year: int = 2018,
                                  bps_end_year: int = 2024) -> TwoLevelAnalysisResult:
        """
        Perform complete two-level food security analysis
        
        Args:
            spatial_analysis_results: Results from spatial analysis (NASA location level)
            bps_start_year: Start year for BPS data
            bps_end_year: End year for BPS data
            
        Returns:
            Complete TwoLevelAnalysisResult
        """
        try:
            self.logger.info("Starting two-level food security analysis")
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
            
            self.logger.info(f"Two-level analysis complete: {len(kecamatan_analyses)} kecamatan → {len(kabupaten_analyses)} kabupaten")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in two-level analysis: {str(e)}")
            raise
    
    def _process_level_1_kecamatan_analyses(self, 
                                          spatial_results: List[Dict[str, Any]]) -> List[KecamatanAnalysis]:
        """Process Level 1: Kecamatan climate-based analyses from NASA locations"""
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
                fsci_analysis = self._convert_spatial_to_fsci_analysis(result)
                
                # Create KecamatanAnalysis with real mapping data
                kecamatan_analysis = KecamatanAnalysis(
                    nasa_location_name=nasa_location_name,
                    kecamatan_name=kecamatan_info.kecamatan_name,
                    kabupaten_name=kecamatan_info.kabupaten_name,
                    area_km2=kecamatan_info.area_km2,
                    area_weight=kecamatan_info.area_weight,
                    fsci_analysis=fsci_analysis
                )
                
                kecamatan_analyses.append(kecamatan_analysis)
                
                self.logger.info(f"Level 1: {nasa_location_name} → {kecamatan_info.kecamatan_name} "
                               f"(FSCI: {fsci_analysis.fsci_score}) [{kecamatan_info.kabupaten_name}]")
            
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
                    
                    # Aggregate FSCI scores using real area weights
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
                        constituent_kecamatan=[k.kecamatan_name for k in kecamatan_group],
                        constituent_nasa_locations=[k.nasa_location_name for k in kecamatan_group],
                        
                        aggregated_fsci_score=aggregated_metrics['fsci'],
                        aggregated_fsci_class=self.kecamatan_analyzer._classify_fsci_score(aggregated_metrics['fsci']),
                        aggregated_pci_score=aggregated_metrics['pci'],
                        aggregated_psi_score=aggregated_metrics['psi'],
                        aggregated_crs_score=aggregated_metrics['crs'],
                        
                        kecamatan_analyses=kecamatan_group,
                        bps_validation=bps_validation,
                        
                        climate_production_correlation=correlation,
                        production_efficiency_score=efficiency_score,
                        climate_potential_rank=0,  # Will be set in ranking phase
                        actual_production_rank=0,  # Will be set in ranking phase
                        performance_gap_category=self._determine_performance_gap_category(
                            aggregated_metrics['fsci'], efficiency_score
                        ),
                        
                        validation_notes=self._generate_validation_notes(
                            aggregated_metrics, bps_validation, correlation
                        )
                    )
                    
                    kabupaten_analyses.append(kabupaten_analysis)
                    
                    self.logger.info(f"Level 2 complete for {kabupaten_name}: "
                                   f"FSCI={aggregated_metrics['fsci']:.1f}, "
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
        """Aggregate kecamatan FSCI scores to kabupaten level using real area weights"""
        try:
            # Use area-weighted aggregation
            weighted_fsci = 0.0
            weighted_pci = 0.0
            weighted_psi = 0.0
            weighted_crs = 0.0
            total_weight = 0.0
            
            for analysis in kecamatan_analyses:
                weight = analysis.area_weight
                
                weighted_fsci += analysis.fsci_analysis.fsci_score * weight
                weighted_pci += analysis.fsci_analysis.pci.pci_score * weight
                weighted_psi += analysis.fsci_analysis.psi.psi_score * weight
                weighted_crs += analysis.fsci_analysis.crs.crs_score * weight
                total_weight += weight
            
            if total_weight == 0:
                self.logger.warning(f"Zero total weight for {kabupaten_name}")
                return {'fsci': 65.0, 'pci': 70.0, 'psi': 65.0, 'crs': 60.0}
            
            aggregated_metrics = {
                'fsci': round(weighted_fsci / total_weight, 2),
                'pci': round(weighted_pci / total_weight, 2),
                'psi': round(weighted_psi / total_weight, 2),
                'crs': round(weighted_crs / total_weight, 2)
            }
            
            self.logger.info(f"Aggregated metrics for {kabupaten_name}: "
                           f"FSCI={aggregated_metrics['fsci']}, weights_sum={total_weight:.3f}")
            
            return aggregated_metrics
                
        except Exception as e:
            self.logger.error(f"Error aggregating metrics for {kabupaten_name}: {str(e)}")
            return {'fsci': 65.0, 'pci': 70.0, 'psi': 65.0, 'crs': 60.0}
    
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
            productions = [record.produksi_padi_ton for record in historical_data.values()]
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
            # Normalize climate score (FSCI) to production scale
            fsci_score = aggregated_metrics['fsci']
            
            # Normalize production to expected range (based on Aceh rice production knowledge)
            # Max expected production per kabupaten ~ 400,000 tons
            max_expected = 400000
            normalized_production = min(100, (bps_validation.latest_production_tons / max_expected) * 100)
            
            # Simple correlation calculation (can be enhanced)
            score_diff = abs(fsci_score - normalized_production)
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
            fsci_score = aggregated_metrics['fsci']
            actual_production = bps_validation.latest_production_tons
            
            # Expected production based on FSCI (rough regional benchmark)
            # FSCI 80 = 350,000 tons, FSCI 60 = 200,000 tons (linear scaling)
            expected_production = ((fsci_score - 50) / 30) * 200000 + 150000
            expected_production = max(100000, expected_production)  # Minimum threshold
            
            # Efficiency calculation
            efficiency = (actual_production / expected_production) * 100
            
            return round(min(200, max(20, efficiency)), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency: {str(e)}")
            return 75.0
    
    def _determine_performance_gap_category(self, fsci_score: float, efficiency_score: float) -> str:
        """Determine performance gap category"""
        gap = efficiency_score - fsci_score
        
        if gap > 15:
            return "overperforming"
        elif gap < -15:
            return "underperforming" 
        else:
            return "aligned"
    
    def _generate_rankings(self, kabupaten_analyses: List[KabupatenAnalysis]) -> tuple:
        """Generate climate and production rankings"""
        # Climate potential ranking
        climate_sorted = sorted(kabupaten_analyses, key=lambda x: x.aggregated_fsci_score, reverse=True)
        climate_ranking = []
        
        for rank, analysis in enumerate(climate_sorted, 1):
            analysis.climate_potential_rank = rank  # Update rank in object
            climate_ranking.append({
                "rank": rank,
                "kabupaten_name": analysis.kabupaten_name,
                "fsci_score": analysis.aggregated_fsci_score,
                "fsci_class": analysis.aggregated_fsci_class.value,
                "constituent_kecamatan": len(analysis.constituent_kecamatan)
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
                "data_years": analysis.bps_validation.data_coverage_years
            })
        
        return climate_ranking, production_ranking
    
    # Helper methods...
    def _convert_spatial_to_fsci_analysis(self, spatial_result: Dict[str, Any]) -> FoodSecurityAnalysis:
        """Convert spatial analysis result to FoodSecurityAnalysis object"""
        # Implementation similar to previous version but updated for your data structure
        from services.food_security_analyzer import (
            ProductionCapacityIndex, ProductionStabilityIndex, ClimateResilienceScore
        )
        
        district = spatial_result.get('district', 'Unknown')
        fsci_score = spatial_result.get('fsci_score', 65.0)
        pci_score = spatial_result.get('pci_score', 75.0)
        psi_score = spatial_result.get('psi_score', 70.0)
        crs_score = spatial_result.get('crs_score', 68.0)
        
        # Create simplified component objects
        pci = ProductionCapacityIndex(
            climate_suitability=pci_score * 0.8,
            land_quality_factor=75.0,
            water_availability_factor=70.0,
            risk_adjustment_factor=25.0,
            pci_score=pci_score,
            pci_class=self.kecamatan_analyzer._classify_pci_score(pci_score)
        )
        
        psi = ProductionStabilityIndex(
            temporal_stability=psi_score * 0.9,
            climate_variability=30.0,
            trend_consistency=75.0,
            anomaly_resilience=70.0,
            psi_score=psi_score,
            psi_class=self.kecamatan_analyzer._classify_psi_score(psi_score)
        )
        
        crs = ClimateResilienceScore(
            temperature_resilience=crs_score * 0.95,
            precipitation_resilience=crs_score * 1.05,
            extreme_weather_resilience=crs_score,
            adaptation_capacity=70.0,
            crs_score=crs_score,
            crs_class=self.kecamatan_analyzer._classify_crs_score(crs_score)
        )
        
        return FoodSecurityAnalysis(
            district_name=district,
            district_code=spatial_result.get('district_code', 'Unknown'),
            pci=pci,
            psi=psi,
            crs=crs,
            fsci_score=fsci_score,
            fsci_class=self.kecamatan_analyzer._classify_fsci_score(fsci_score),
            investment_recommendation=spatial_result.get('investment_recommendation', 'Further evaluation needed'),
            priority_ranking=spatial_result.get('priority_ranking', 0),
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
        
        fsci_score = aggregated_metrics['fsci']
        
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
        """Generate comprehensive cross-level insights"""
        try:
            insights = {
                "methodology_validation": {},
                "climate_production_alignment": {},
                "spatial_variability": {},
                "policy_recommendations": {}
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
                kecamatan_fsci_scores = [k.fsci_analysis.fsci_score for k in analysis.kecamatan_analyses]
                
                if len(kecamatan_fsci_scores) > 1:
                    fsci_variance = np.var(kecamatan_fsci_scores)
                    fsci_range = max(kecamatan_fsci_scores) - min(kecamatan_fsci_scores)
                    
                    insights["spatial_variability"][analysis.kabupaten_name] = {
                        "kecamatan_fsci_range": f"{min(kecamatan_fsci_scores):.1f}-{max(kecamatan_fsci_scores):.1f}",
                        "fsci_variance": round(fsci_variance, 2),
                        "internal_variability": "high" if fsci_variance > 50 else "moderate" if fsci_variance > 15 else "low",
                        "largest_contributor": max(analysis.kecamatan_analyses, key=lambda x: x.area_weight).kecamatan_name
                    }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating cross-level insights: {str(e)}")
            return {}
    
    def _generate_methodology_summary(self) -> str:
        """Generate methodology summary for reporting"""
        return ("Two-level hybrid food security analysis: (1) Kecamatan-level climate suitability analysis "
                "using NASA POWER data at 10 locations mapped to administrative boundaries via GeoJSON, "
                "(2) Area-weighted aggregation to kabupaten level with BPS rice production validation (2018-2024). "
                "Climate potential assessed at micro-scale, validated against macro-scale actual production data "
                "using Indonesian administrative hierarchy.")