from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime
from services.geojson_loader import GeojsonLoader
from services.climate_data_service import ClimateDataService
from services.rice_analyzer_service import RiceAnalyzer
from services.food_security_analyzer import FoodSecurityAnalyzer
from services.two_level_food_security_analyzer import TwoLevelFoodSecurityAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint for two-level analysis
two_level_api = Blueprint('two_level_api', __name__, url_prefix='/api/v1/two-level')

@two_level_api.route('/analysis', methods=['GET'])
def perform_two_level_analysis():
    """
    Main endpoint for two-level food security analysis
    Integrates kecamatan-level climate analysis with kabupaten-level BPS validation
    """
    try:
        logger.info("🍚🔬 Starting Two-Level Food Security Analysis")
        
        # Get parameters with BPS-aligned defaults
        districts = request.args.get('districts', 'all')
        bps_start_year = int(request.args.get('bps_start_year', 2018))
        bps_end_year = int(request.args.get('bps_end_year', 2024))
        
        # AUTO-ALIGN: NASA period matches BPS availability (2018-2024)
        year_start = int(request.args.get('year_start', bps_start_year))  # Default to BPS start
        year_end = int(request.args.get('year_end', bps_end_year))      # Default to BPS end
        
        season = request.args.get('season', 'all')
        aggregation = request.args.get('aggregation', 'mean')
        
        logger.info(f"📊 AUTO-ALIGNED Parameters:")
        logger.info(f"   NASA Climate: {year_start}-{year_end}")
        logger.info(f"   BPS Production: {bps_start_year}-{bps_end_year}")
        logger.info(f"   Temporal overlap: {max(year_start, bps_start_year)}-{min(year_end, bps_end_year)}")
        
        # Initialize services
        db = current_app.config['MONGO_DB']
        geojson_loader = GeojsonLoader()
        climate_service = ClimateDataService(db)
        rice_analyzer = RiceAnalyzer()
        food_security_analyzer = FoodSecurityAnalyzer()
        two_level_analyzer = TwoLevelFoodSecurityAnalyzer()
        
        # Step 1: Get spatial analysis results (Level 1 data)
        logger.info("🌍 Step 1: Performing spatial food security analysis...")
        spatial_results = _perform_spatial_food_security_analysis(
            geojson_loader, climate_service, rice_analyzer, food_security_analyzer,
            year_start, year_end, season, aggregation
        )
        
        if not spatial_results:
            return jsonify({
                "message": "No spatial analysis results available",
                "error": "Failed to generate Level 1 kecamatan analysis"
            }), 500
        
        logger.info(f"✅ Level 1 complete: {len(spatial_results)} NASA locations analyzed")
        
        # Step 2: Perform two-level analysis
        logger.info("🏛️ Step 2: Performing two-level analysis with BPS validation...")
        two_level_result = two_level_analyzer.perform_two_level_analysis(
            spatial_analysis_results=spatial_results,
            bps_start_year=bps_start_year,
            bps_end_year=bps_end_year
        )
        
        logger.info(f"✅ Two-level analysis complete: "
                   f"{two_level_result.level_1_kecamatan_count} kecamatan → "
                   f"{two_level_result.level_2_kabupaten_count} kabupaten")
        
        # Step 3: Create comprehensive response with temporal metadata
        response = _create_two_level_response(
            two_level_result, year_start, year_end, bps_start_year, bps_end_year
        )
        
        # Add temporal alignment info to response
        response["temporal_alignment"] = {
            "nasa_period": f"{year_start}-{year_end}",
            "bps_period": f"{bps_start_year}-{bps_end_year}",
            "alignment_method": "auto_aligned_to_bps",
            "overlap_years": max(0, min(year_end, bps_end_year) - max(year_start, bps_start_year) + 1),
            "correlation_quality": "high" if year_start >= bps_start_year and year_end <= bps_end_year else "moderate"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Two-level analysis failed: {str(e)}")
        return jsonify({
            "message": "Two-level food security analysis failed",
            "error": str(e),
            "endpoint": "/api/v1/two-level/analysis"
        }), 500

@two_level_api.route('/summary', methods=['GET'])
def get_two_level_summary():
    """
    Quick summary endpoint for two-level analysis capabilities
    """
    try:
        from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
        
        mapping_service = KecamatanKabupatenMappingService()
        
        # Get mapping validation
        validation = mapping_service.validate_mapping_consistency()
        
        return jsonify({
            "analysis_type": "Two-Level Hybrid Food Security Analysis",
            "description": "Climate-based kecamatan analysis validated with BPS kabupaten production data",
            "methodology": {
                "level_1": "Kecamatan climate suitability analysis (NASA POWER data)",
                "level_2": "Kabupaten aggregation with BPS production validation",
                "aggregation_method": "Area-weighted average using GeoJSON geometry",
                "validation_approach": "Climate potential vs actual production correlation"
            },
            "coverage": {
                "total_kabupaten": validation["total_kabupaten"],
                "total_kecamatan": validation["total_kecamatan"],
                "nasa_locations": validation["total_nasa_locations"],
                "bps_compatible": True
            },
            "data_sources": {
                "climate": "NASA POWER (2005-2023)",
                "production": "BPS Statistics (2018-2024)",
                "boundaries": "GeoJSON with area calculations",
                "temporal": "20-year climate analysis"
            },
            "outputs": {
                "kecamatan_level": ["FSCI", "PCI", "PSI", "CRS", "Investment recommendations"],
                "kabupaten_level": ["Aggregated FSCI", "BPS validation", "Production efficiency", "Rankings"],
                "cross_level": ["Climate-production correlation", "Performance gaps", "Policy insights"]
            },
            "available_endpoints": {
                "full_analysis": "/api/v1/two-level/analysis",
                "kabupaten_summary": "/api/v1/two-level/kabupaten-summary",
                "validation_report": "/api/v1/two-level/validation",
                "mapping_info": "/api/v1/two-level/mapping"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in two-level summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

@two_level_api.route('/kabupaten-summary', methods=['GET'])
def get_kabupaten_summary():
    """
    Get summary of kabupaten-level aggregation without full analysis
    """
    try:
        from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
        
        mapping_service = KecamatanKabupatenMappingService()
        
        # Get detailed mapping summary
        summary = mapping_service.get_detailed_mapping_summary()
        
        return jsonify({
            "message": "Kabupaten aggregation summary",
            "data": summary,
            "aggregation_ready": True,
            "bps_integration_ready": True
        }), 200
        
    except Exception as e:
        logger.error(f"Error in kabupaten summary: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@two_level_api.route('/bps-historical/<kabupaten>', methods=['GET'])
def get_bps_historical_data(kabupaten: str):
    """
    Get BPS historical production data for specific kabupaten
    Integrates with existing BPS service
    """
    try:
        from services.bps_api_service import BPSApiService
        
        start_year = int(request.args.get('start_year', 2018))
        end_year = int(request.args.get('end_year', 2024))
        
        logger.info(f"BPS historical request for {kabupaten}: {start_year}-{end_year}")
        
        # Initialize BPS service
        bps_service = BPSApiService()
        
        # Fetch historical data
        historical_data = bps_service.fetch_kabupaten_historical_data(
            kabupaten, start_year, end_year
        )
        
        if not historical_data:
            return jsonify({
                "message": f"No BPS historical data found for {kabupaten}",
                "error": "Data not available",
                "available_kabupaten": bps_service.target_kabupaten
            }), 404
        
        # Format response for two-level compatibility
        yearly_production = []
        for year, record in historical_data.items():
            yearly_production.append({
                "year": year,
                "production_tons": record.produksi_padi_ton,
                "luas_tanam_ha": getattr(record, 'luas_tanam_ha', 0),
                "luas_panen_ha": getattr(record, 'luas_panen_ha', 0),
                "produktivitas_ton_ha": getattr(record, 'produktivitas_ton_ha', 0),
                "harvest_success_rate": getattr(record, 'harvest_success_rate', 0)
            })
        
        # Calculate production statistics
        production_values = [record.produksi_padi_ton for record in historical_data.values()]
        
        return jsonify({
            "kabupaten_name": kabupaten,
            "bps_compatible_name": kabupaten,
            "data_period": f"{start_year}-{end_year}",
            "data_coverage_years": len(historical_data),
            "yearly_production_data": yearly_production,
            "production_statistics": {
                "total_production_tons": sum(production_values),
                "average_annual_production": sum(production_values) / len(production_values),
                "max_production": max(production_values),
                "min_production": min(production_values),
                "production_volatility_percent": (
                    (max(production_values) - min(production_values)) / 
                    (sum(production_values) / len(production_values)) * 100
                )
            },
            "trend_analysis": {
                "overall_change_percent": (
                    (production_values[-1] - production_values[0]) / production_values[0] * 100
                    if len(production_values) > 1 else 0
                ),
                "most_productive_year": max(historical_data.keys(), key=lambda y: historical_data[y].produksi_padi_ton),
                "least_productive_year": min(historical_data.keys(), key=lambda y: historical_data[y].produksi_padi_ton)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in BPS historical endpoint: {str(e)}")
        return jsonify({
            "message": "Failed to fetch BPS historical data",
            "error": str(e)
        }), 500

@two_level_api.route('/validation', methods=['GET'])
def get_validation_report():
    """
    Get comprehensive validation report for two-level system
    """
    try:
        from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
        from services.area_weight_service import AreaWeightService
        
        mapping_service = KecamatanKabupatenMappingService()
        area_service = AreaWeightService()
        
        # Get all validation data
        mapping_validation = mapping_service.validate_mapping_consistency()
        area_validation = area_service.validate_area_weights()
        
        return jsonify({
            "validation_report": {
                "mapping_consistency": mapping_validation,
                "area_weights": area_validation,
                "system_readiness": {
                    "level_1_ready": True,
                    "level_2_ready": True,
                    "bps_integration_ready": True,
                    "geojson_loaded": area_validation["total_kecamatan"] > 0
                }
            },
            "validation_timestamp": datetime.now().isoformat(),
            "recommendation": "System ready for two-level analysis" if area_validation["total_kecamatan"] > 0 else "Check GeoJSON file path"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in validation report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@two_level_api.route('/mapping', methods=['GET'])
def get_mapping_info():
    """
    Get NASA location to kecamatan to kabupaten mapping information
    """
    try:
        from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService
        
        mapping_service = KecamatanKabupatenMappingService()
        
        # Get detailed mapping
        detailed_summary = mapping_service.get_detailed_mapping_summary()
        
        return jsonify({
            "mapping_information": detailed_summary,
            "nasa_location_count": detailed_summary["service_info"]["total_nasa_locations"],
            "kecamatan_count": detailed_summary["service_info"]["total_kecamatan"], 
            "kabupaten_count": detailed_summary["service_info"]["total_kabupaten"],
            "data_source": detailed_summary["service_info"]["area_data_source"]
        }), 200
        
    except Exception as e:
        logger.error(f"Error in mapping info: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Helper functions
def _perform_spatial_food_security_analysis(geojson_loader, climate_service, rice_analyzer, 
                                           food_security_analyzer, year_start, year_end, 
                                           season, aggregation):
    """
    Perform spatial food security analysis (reuse logic from existing spatial_api.py)
    Returns list of analysis results for each NASA location
    """
    try:
        # Load kecamatan spatial data
        kecamatan_gdf = geojson_loader.load_kecamatan_geojson()
        
        if kecamatan_gdf is None or len(kecamatan_gdf) == 0:
            return []
        
        analysis_results = []
        
        for idx, kecamatan in kecamatan_gdf.iterrows():
            nasa_match = kecamatan['nasa_match']
            district_name = kecamatan.get('NAME_3', nasa_match)
            
            try:
                # Get climate data
                collection_name = climate_service.get_collection_name_for_nasa_location(nasa_match)
                if not collection_name:
                    continue
                
                climate_data = climate_service.load_climate_data(
                    collection_name, year_start, year_end
                )
                
                if climate_data is None or len(climate_data) == 0:
                    continue
                
                # Apply filters
                if season != 'all':
                    climate_data = climate_service.apply_seasonal_filter(climate_data, season)
                
                # Get base suitability
                aggregated_data = climate_service.aggregate_climate_data(climate_data, aggregation)
                suitability_result = rice_analyzer.calculate_suitability_score(aggregated_data)
                
                # Convert climate data for temporal analysis
                if hasattr(climate_data, 'to_dict'):
                    climate_records = climate_data.to_dict('records')
                else:
                    climate_records = climate_data
                
                # Food security analysis
                district_data = kecamatan.to_dict()
                fsci_analysis = food_security_analyzer.analyze_food_security(
                    district_data=district_data,
                    climate_time_series=climate_records,
                    base_suitability_score=suitability_result['score']
                )
                
                # Create result record for two-level analyzer
                analysis_result = {
                    'district': nasa_match,  # NASA location name
                    'district_code': kecamatan.get('GID_3', 'Unknown'),
                    'fsci_score': fsci_analysis.fsci_score,
                    'pci_score': fsci_analysis.pci.pci_score,
                    'psi_score': fsci_analysis.psi.psi_score,
                    'crs_score': fsci_analysis.crs.crs_score,
                    'investment_recommendation': fsci_analysis.investment_recommendation,
                    'priority_ranking': 0,  # Will be set later
                    'suitability_score': suitability_result['score'],
                    'classification': suitability_result['classification']
                }
                
                analysis_results.append(analysis_result)
                
            except Exception as e:
                logger.error(f"Error analyzing {nasa_match}: {str(e)}")
                continue
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in spatial food security analysis: {str(e)}")
        return []

def _create_two_level_response(two_level_result, year_start, year_end, bps_start_year, bps_end_year):
    """
    Create comprehensive API response from two-level analysis result
    """
    try:
        # Convert dataclass objects to dictionaries for JSON serialization
        kecamatan_data = []
        for k_analysis in two_level_result.kecamatan_analyses:
            kecamatan_data.append({
                "nasa_location_name": k_analysis.nasa_location_name,
                "kecamatan_name": k_analysis.kecamatan_name,
                "kabupaten_name": k_analysis.kabupaten_name,
                "area_km2": k_analysis.area_km2,
                "area_weight": k_analysis.area_weight,
                "fsci_score": k_analysis.fsci_analysis.fsci_score,
                "fsci_class": k_analysis.fsci_analysis.fsci_class.value,
                "pci_score": k_analysis.fsci_analysis.pci.pci_score,
                "psi_score": k_analysis.fsci_analysis.psi.psi_score,
                "crs_score": k_analysis.fsci_analysis.crs.crs_score,
                "investment_recommendation": k_analysis.fsci_analysis.investment_recommendation
            })
        
        kabupaten_data = []
        for kb_analysis in two_level_result.kabupaten_analyses:
            kabupaten_data.append({
                "kabupaten_name": kb_analysis.kabupaten_name,
                "bps_compatible_name": kb_analysis.bps_compatible_name,
                "total_area_km2": kb_analysis.total_area_km2,
                "constituent_kecamatan": kb_analysis.constituent_kecamatan,
                "constituent_nasa_locations": kb_analysis.constituent_nasa_locations,
                "aggregated_fsci_score": kb_analysis.aggregated_fsci_score,
                "aggregated_fsci_class": kb_analysis.aggregated_fsci_class.value,
                "aggregated_pci_score": kb_analysis.aggregated_pci_score,
                "aggregated_psi_score": kb_analysis.aggregated_psi_score,
                "aggregated_crs_score": kb_analysis.aggregated_crs_score,
                "latest_production_tons": kb_analysis.bps_validation.latest_production_tons,
                "average_production_tons": kb_analysis.bps_validation.average_production_tons,
                "production_trend": kb_analysis.bps_validation.production_trend,
                "data_coverage_years": kb_analysis.bps_validation.data_coverage_years,
                "climate_production_correlation": kb_analysis.climate_production_correlation,
                "production_efficiency_score": kb_analysis.production_efficiency_score,
                "climate_potential_rank": kb_analysis.climate_potential_rank,
                "actual_production_rank": kb_analysis.actual_production_rank,
                "performance_gap_category": kb_analysis.performance_gap_category,
                "validation_notes": kb_analysis.validation_notes
            })
        
        return {
            "type": "TwoLevelFoodSecurityAnalysis",
            "metadata": {
                "analysis_timestamp": two_level_result.analysis_timestamp,
                "analysis_period": two_level_result.analysis_period,
                "bps_data_period": two_level_result.bps_data_period,
                "methodology_summary": two_level_result.methodology_summary,
                "level_1_kecamatan_count": two_level_result.level_1_kecamatan_count,
                "level_2_kabupaten_count": two_level_result.level_2_kabupaten_count,
                "data_sources": {
                    "climate_analysis": f"NASA POWER ({year_start}-{year_end})",
                    "production_validation": f"BPS Statistics ({bps_start_year}-{bps_end_year})",
                    "spatial_boundaries": "GeoJSON with area calculations",
                    "administrative_mapping": "Kecamatan-Kabupaten hierarchy"
                }
            },
            "level_1_kecamatan_analysis": {
                "description": "Climate-based food security analysis at kecamatan level",
                "analysis_count": len(kecamatan_data),
                "data": kecamatan_data
            },
            "level_2_kabupaten_analysis": {
                "description": "Aggregated analysis with BPS production validation",
                "analysis_count": len(kabupaten_data),
                "data": kabupaten_data
            },
            "rankings": {
                "climate_potential_ranking": two_level_result.kabupaten_climate_ranking,
                "actual_production_ranking": two_level_result.kabupaten_production_ranking
            },
            "cross_level_insights": two_level_result.cross_level_insights,
            "summary_statistics": _calculate_summary_statistics(kabupaten_data),
            "recommendations": _generate_policy_recommendations(kabupaten_data)
        }
        
    except Exception as e:
        logger.error(f"Error creating two-level response: {str(e)}")
        raise

def _calculate_summary_statistics(kabupaten_data):
    """Calculate summary statistics from kabupaten analysis"""
    if not kabupaten_data:
        return {}
    
    fsci_scores = [k["aggregated_fsci_score"] for k in kabupaten_data]
    production_totals = [k["latest_production_tons"] for k in kabupaten_data]
    correlations = [k["climate_production_correlation"] for k in kabupaten_data]
    
    return {
        "average_fsci_score": round(sum(fsci_scores) / len(fsci_scores), 2),
        "total_production_tons": round(sum(production_totals), 0),
        "average_climate_production_correlation": round(sum(correlations) / len(correlations), 3),
        "high_potential_kabupaten": len([k for k in kabupaten_data if k["aggregated_fsci_score"] >= 70]),
        "underperforming_kabupaten": len([k for k in kabupaten_data if k["performance_gap_category"] == "underperforming"]),
        "data_coverage": {
            "full_bps_coverage": len([k for k in kabupaten_data if k["data_coverage_years"] >= 5]),
            "limited_bps_coverage": len([k for k in kabupaten_data if k["data_coverage_years"] < 5])
        }
    }

def _generate_policy_recommendations(kabupaten_data):
    """Generate policy recommendations based on analysis"""
    recommendations = []
    
    # High potential areas
    high_potential = [k for k in kabupaten_data if k["aggregated_fsci_score"] >= 75]
    if high_potential:
        recommendations.append({
            "category": "Investment Priority",
            "target": [k["kabupaten_name"] for k in high_potential],
            "recommendation": "Prioritize infrastructure development and technology adoption",
            "rationale": "High climate potential with strong production capacity"
        })
    
    # Underperforming areas
    underperforming = [k for k in kabupaten_data if k["performance_gap_category"] == "underperforming"]
    if underperforming:
        recommendations.append({
            "category": "Performance Gap",
            "target": [k["kabupaten_name"] for k in underperforming],
            "recommendation": "Investigate production constraints and implement targeted interventions",
            "rationale": "Climate potential not fully realized in actual production"
        })
    
    # Low correlation areas
    low_correlation = [k for k in kabupaten_data if k["climate_production_correlation"] < 0.4]
    if low_correlation:
        recommendations.append({
            "category": "System Efficiency",
            "target": [k["kabupaten_name"] for k in low_correlation],
            "recommendation": "Improve agricultural systems alignment with climate conditions",
            "rationale": "Low correlation between climate potential and actual production"
        })
    
    return recommendations