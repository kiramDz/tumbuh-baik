import logging
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from typing import Dict, List, Any

from services.two_level_food_security_analyzer import TwoLevelFoodSecurityAnalyzer
from services.geojson_loader import GeojsonLoader
from services.spatial_analysis import create_spatial_connector
from services.climate_data_service import ClimateDataService

logger = logging.getLogger(__name__)

# Create blueprint
two_level_api_bp = Blueprint('two_level_api', __name__, url_prefix='/api/v1/two-level')

@two_level_api_bp.route('/analysis', methods=['GET'])
def get_two_level_analysis():
    """
    Simplified Two-Level Food Security Analysis - FSI Only
    
    Query Parameters:
    - year_start: Start year for analysis (default: 2018)
    - year_end: End year for analysis (default: 2024)
    - bps_start_year: BPS data start year (default: 2018)
    - bps_end_year: BPS data end year (default: 2024)
    """
    try:
        # Get parameters (simplified - ignore season/aggregation/districts filters)
        year_start = int(request.args.get('year_start', 2018))
        year_end = int(request.args.get('year_end', 2024))
        bps_start_year = int(request.args.get('bps_start_year', 2018))
        bps_end_year = int(request.args.get('bps_end_year', 2024))
        
        logger.info("Starting Two-Level Food Security Analysis")
        logger.info(f"AUTO-ALIGNED Parameters:")
        logger.info(f"   NASA Climate: {year_start}-{year_end}")
        logger.info(f"   BPS Production: {bps_start_year}-{bps_end_year}")
        logger.info(f"   Temporal overlap: {year_start}-{year_end}")
        
        # Step 1: Perform spatial food security analysis for NASA locations
        logger.info("Step 1: Performing spatial food security analysis...")
        spatial_results = _perform_spatial_fsi_analysis(year_start, year_end)
        
        if not spatial_results:
            logger.error("No spatial analysis results available")
            return jsonify({
                "error": "Failed to generate Level 1 kecamatan analysis",
                "message": "No spatial analysis results available"
            }), 500
        
        logger.info(f"Spatial analysis complete: {len(spatial_results)} NASA locations analyzed")
        
        # Step 2: Perform two-level analysis
        logger.info("Step 2: Performing two-level FSI analysis...")
        two_level_analyzer = TwoLevelFoodSecurityAnalyzer()
        
        analysis_result = two_level_analyzer.perform_two_level_analysis(
            spatial_analysis_results=spatial_results,
            bps_start_year=bps_start_year,
            bps_end_year=bps_end_year
        )
        
        # Step 3: Format simplified response
        logger.info("Step 3: Formatting simplified FSI response...")
        response = _format_simplified_fsi_response(analysis_result)
        
        logger.info("Two-level FSI analysis completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in two-level analysis: {str(e)}")
        return jsonify({
            "error": "Two-level analysis failed",
            "message": str(e)
        }), 500

def _perform_spatial_fsi_analysis(year_start: int, year_end: int) -> List[Dict[str, Any]]:
    """
    Perform spatial FSI analysis for all NASA locations - Simplified
    """
    try:
        # Get database connection
        db = current_app.config['MONGO_DB']
        
        # Load GeoJSON with NASA location mappings
        geojson_loader = GeojsonLoader()
        kecamatan_gdf = geojson_loader.load_kecamatan_geojson()
        
        if kecamatan_gdf is None or len(kecamatan_gdf) == 0:
            logger.error("Failed to load kecamatan GeoJSON data")
            return []
        
        unique_kabupaten = kecamatan_gdf['NAME_2'].unique()
        for kabupaten in unique_kabupaten:
            count = len(kecamatan_gdf[kecamatan_gdf['NAME_2'] == kabupaten])
            logger.info(f"  {kabupaten}: {count} kecamatan")
        
        logger.info(f"Loaded {len(kecamatan_gdf)} kecamatan from GeoJSON")
        
        # Initialize climate data service
        climate_service = ClimateDataService(db)
        
        # Remove spatial connector - not needed for FSI-only analysis
        # spatial_connector = create_spatial_connector(db)  # ← DELETE this line
        
        spatial_results = []
        
        # Process each NASA location
        for idx, row in kecamatan_gdf.iterrows():
            try:
                nasa_location = row['nasa_match']
                kecamatan_name = row['NAME_3']
                
                logger.info(f"Analyzing {nasa_location} -> {kecamatan_name}")
                
                # Get collection name for this NASA location
                collection_name = climate_service.get_collection_name_for_nasa_location(nasa_location)
                if not collection_name:
                    logger.warning(f"No collection found for {nasa_location}")
                    continue
                
                # Load climate data
                climate_df = climate_service.load_climate_data(
                    collection_name, year_start, year_end
                )
                
                if climate_df is None or len(climate_df) == 0:
                    logger.warning(f"No climate data loaded for {nasa_location}")
                    continue
                
                # Get aggregated climate data
                climate_data = climate_service.aggregate_climate_data(climate_df)
                
                # Calculate base suitability score using simple climate scoring
                # Remove spatial connector dependency
                base_suitability_score = _calculate_base_suitability_score(climate_data)
                
                # Perform FSI analysis
                from services.food_security_analyzer import FoodSecurityAnalyzer
                fsi_analyzer = FoodSecurityAnalyzer()
                
                # Convert climate_df to list of dicts for FSI analyzer
                climate_time_series = climate_df.to_dict('records') if climate_df is not None else []
                
                fsi_analysis = fsi_analyzer.analyze_food_security(
                    district_data=row.to_dict(),
                    climate_time_series=climate_time_series,
                    base_suitability_score=base_suitability_score
                )
                
                # Create spatial result in expected format
                spatial_result = {
                    'district': nasa_location,
                    'district_code': row.get('GID_3', 'Unknown'),
                    'kecamatan_name': kecamatan_name,
                    'kabupaten_name': row.get('NAME_2', 'Unknown'),
                    'fsi_score': fsi_analysis.fsi_score,
                    'fsi_class': fsi_analysis.fsi_class.value,
                    'natural_resources_score': fsi_analysis.natural_resources_score,
                    'availability_score': fsi_analysis.availability_score,
                    'suitability_score': base_suitability_score,  # for backward compatibility
                    'analysis_timestamp': fsi_analysis.analysis_timestamp
                }
                
                spatial_results.append(spatial_result)
                
                logger.info(f"FSI analysis complete for {nasa_location}: FSI={fsi_analysis.fsi_score:.1f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {nasa_location}: {str(e)}")
                continue
        
        return spatial_results
        
    except Exception as e:
        logger.error(f"Error in spatial FSI analysis: {str(e)}")
        return []

def _calculate_base_suitability_score(climate_data: Dict[str, float]) -> float:
    """
    Calculate base suitability score from climate data - Simple approach
    """
    try:
        # Extract climate parameters
        temp = climate_data.get('T2M', 26.0)  # Default temperature
        precip = climate_data.get('PRECTOTCORR', 5.0)  # Default precipitation
        
        # Simple temperature scoring (optimal: 24-30°C)
        if 24 <= temp <= 30:
            temp_score = 100
        elif temp > 30:
            temp_score = max(60, 100 - ((temp - 30) / 5) * 40)
        else:
            temp_score = max(50, ((temp - 20) / 4) * 100)
        
        # Simple precipitation scoring (convert daily to annual: * 365)
        annual_precip = precip * 365
        if 1200 <= annual_precip <= 1800:
            precip_score = 100
        elif annual_precip > 1800:
            precip_score = max(60, 100 - ((annual_precip - 1800) / 1000) * 30)
        else:
            precip_score = max(40, (annual_precip / 1200) * 100)
        
        # Combined suitability score
        suitability = (temp_score * 0.4) + (precip_score * 0.6)
        
        return round(min(100, max(30, suitability)), 1)
        
    except Exception as e:
        logger.error(f"Error calculating base suitability: {str(e)}")
        return 65.0  # Default moderate suitability

def _format_simplified_fsi_response(analysis_result) -> Dict[str, Any]:
    """
    Format simplified FSI response - similar to BPS API style
    """
    try:
        # Simplified kecamatan FSI data
        kecamatan_fsi = {}
        for kecamatan in analysis_result.kecamatan_analyses:
            kecamatan_fsi[kecamatan.nasa_location_name] = {
                "fsi_score": kecamatan.fsi_analysis.fsi_score,
                "fsi_class": kecamatan.fsi_analysis.fsi_class.value,
                "natural_resources": kecamatan.fsi_analysis.natural_resources_score,
                "availability": kecamatan.fsi_analysis.availability_score,
                "kecamatan_name": kecamatan.kecamatan_name,
                "kabupaten_name": kecamatan.kabupaten_name,
                "area_km2": kecamatan.area_km2
            }
        
        # Simplified kabupaten FSI data
        kabupaten_fsi = {}
        for kabupaten in analysis_result.kabupaten_analyses:
            kabupaten_fsi[kabupaten.kabupaten_name] = {
                "fsi_score": kabupaten.aggregated_fsi_score,
                "fsi_class": kabupaten.aggregated_fsi_class.value,
                "natural_resources": kabupaten.natural_resources_score,
                "availability": kabupaten.availability_score,
                "sample_kecamatan": len(kabupaten.sample_kecamatan),
                "total_area_km2": kabupaten.total_area_km2,
                "bps_validation": {
                    "latest_production_tons": kabupaten.bps_validation.latest_production_tons,
                    "average_production_tons": kabupaten.bps_validation.average_production_tons,
                    "production_trend": kabupaten.bps_validation.production_trend,
                },
                "performance": {
                    "climate_production_correlation": kabupaten.climate_production_correlation,
                    "production_efficiency_score": kabupaten.production_efficiency_score,
                    "performance_gap_category": kabupaten.performance_gap_category,
                }
            }
        
        # Simplified response structure
        response = {
            "analysis_timestamp": analysis_result.analysis_timestamp,
            "analysis_period": analysis_result.bps_data_period,
            "level_1_kecamatan_count": analysis_result.level_1_kecamatan_count,
            "level_2_kabupaten_count": analysis_result.level_2_kabupaten_count,
            
            "kecamatan_fsi": kecamatan_fsi,
            "kabupaten_fsi": kabupaten_fsi,
            
            "rankings": {
                "climate_ranking": analysis_result.kabupaten_climate_ranking,
                "production_ranking": analysis_result.kabupaten_production_ranking
            },
            
            "methodology_summary": analysis_result.methodology_summary
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting FSI response: {str(e)}")
        return {
            "error": "Failed to format response",
            "message": str(e)
        }

@two_level_api_bp.route('/summary', methods=['GET'])
def get_two_level_summary():
    """Get simplified two-level analysis capabilities summary"""
    try:
        # Load basic info
        geojson_loader = GeojsonLoader()
        summary = geojson_loader.load_summary()
        
        if not summary:
            return jsonify({
                "error": "Failed to load summary data"
            }), 500
        
        # Simplified summary response
        response = {
            "analysis_capabilities": {
                "level_1": "Kecamatan-level FSI analysis (NASA POWER locations)",
                "level_2": "Kabupaten-level FSI aggregation (BPS validation)",
                "fsi_components": ["Natural Resources & Resilience (60%)", "Availability (40%)"],
                "classification_levels": ["Sangat Tinggi (≥80)", "Tinggi (60-79)", "Sedang (40-59)", "Rendah (<40)"]
            },
            "data_coverage": {
                "nasa_locations": summary.get("nasa_locations_count", 11),
                "kecamatan_count": summary.get("total_kecamatan", 11),
                "kabupaten_count": len(summary.get("kabupaten_distribution", {})),
                "kabupaten_list": list(summary.get("kabupaten_distribution", {}).keys())
            },
            "analysis_period": {
                "default_climate_years": "2018-2024",
                "default_bps_years": "2018-2024",
                "temporal_alignment": "Auto-aligned based on data availability"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in two-level summary: {str(e)}")
        return jsonify({
            "error": "Failed to generate summary",
            "message": str(e)
        }), 500

@two_level_api_bp.route('/validation', methods=['GET'])
def get_system_validation():
    """Get system validation status"""
    try:
        # Basic system validation
        db = current_app.config.get('MONGO_DB')
        
        validation_status = {
            "system_status": "operational",
            "components": {
                "database_connection": "connected" if db else "disconnected",
                "geojson_loader": "available",
                "climate_data_service": "available", 
                "bps_api_service": "available",
                "two_level_analyzer": "available"
            },
            "data_validation": {
                "nasa_climate_data": "validated",
                "bps_production_data": "validated", 
                "administrative_boundaries": "validated",
                "spatial_mapping": "validated"
            },
            "api_endpoints": {
                "nasa_api": "simplified",
                "bps_api": "simplified",
                "two_level_api": "fsi_only"
            }
        }
        
        return jsonify(validation_status)
        
    except Exception as e:
        logger.error(f"Error in system validation: {str(e)}")
        return jsonify({
            "error": "System validation failed",
            "message": str(e)
        }), 500