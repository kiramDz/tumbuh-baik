from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime
from services.geojson_loader import GeojsonLoader
from services.climate_data_service import ClimateDataService  
from services.rice_analyzer_service import RiceAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint with /api prefix to match Next.js structure
spatial_api = Blueprint('spatial_api', __name__, url_prefix='/api/v1')

@spatial_api.route('/spatial-map', methods=['GET'])
def get_climate_potential_analysis():
    """
    Main endpoint for climate potential analysis
    Replaces Next.js GET /api/v1/spatial-map
    """
    try:
        logger.info("Starting climate spatial analysis request")

        # Get query parameters (same as Next.js implementation)
        districts = request.args.get('districts', 'all')
        parameters = request.args.get('parameters', 'all')
        year_start = int(request.args.get('year_start', 2004))
        year_end = int(request.args.get('year_end', 2023))
        season = request.args.get('season', 'all')
        aggregation = request.args.get('aggregation', 'mean')

        logger.info(f"Analysis parameters: districts={districts}, "
                   f"year_range={year_start}-{year_end}, season={season}, aggregation={aggregation}")

        # Initialize services
        db = current_app.config['MONGO_DB']
        geojson_loader = GeojsonLoader()
        climate_service = ClimateDataService(db)
        rice_analyzer = RiceAnalyzer()
        
        # Step 1: Load kecamatan spatial data
        logger.info("Loading kecamatan spatial data...")
        kecamatan_gdf = geojson_loader.load_kecamatan_geojson()
        
        if kecamatan_gdf is None or len(kecamatan_gdf) == 0:
            return jsonify({
                "message": "Invalid spatial analysis response",
                "error": "Missing GeoJSON features"
            }), 500
        
        # Step 2: Analyze each kecamatan
        analysis_results = {}
        
        for idx, kecamatan in kecamatan_gdf.iterrows():
            nasa_match = kecamatan['nasa_match']
            
            # Get collection name for this NASA location
            collection_name = climate_service.get_collection_name_for_nasa_location(nasa_match)
            
            if not collection_name:
                logger.warning(f"No collection found for {nasa_match}")
                continue
                
            try:
                # Load climate data for the specified period
                climate_data = climate_service.load_climate_data(
                    collection_name, year_start, year_end
                )
                
                if climate_data is None or len(climate_data) == 0:
                    logger.warning(f"No climate data for {nasa_match}")
                    continue
                
                # Apply seasonal filter if specified
                if season != 'all':
                    climate_data = climate_service.apply_seasonal_filter(
                        climate_data, season
                    )
                
                # Aggregate climate data using specified method
                aggregated_data = climate_service.aggregate_climate_data(
                    climate_data, aggregation
                )
                
                # Calculate rice suitability score
                suitability_result = rice_analyzer.calculate_suitability_score(aggregated_data)
                
                analysis_results[nasa_match] = {
                    'climate_data': aggregated_data,
                    'suitability': suitability_result
                }
                
                logger.info(f"Analyzed {nasa_match}: {suitability_result['classification']}")
                
            except Exception as e:
                logger.error(f"Error analyzing {nasa_match}: {str(e)}")
                continue
        
        # Step 3: Join results to GeoDataFrame
        for idx, kecamatan in kecamatan_gdf.iterrows():
            nasa_match = kecamatan['nasa_match']
            
            if nasa_match in analysis_results:
                result = analysis_results[nasa_match]
                suitability = result['suitability']
                climate = result['climate_data']
                
                # Add suitability properties
                kecamatan_gdf.at[idx, 'suitability_score'] = suitability['score']
                kecamatan_gdf.at[idx, 'classification'] = suitability['classification']
                kecamatan_gdf.at[idx, 'confidence_level'] = suitability.get('confidence_level', 'medium')
                
                # Add component scores
                component_scores = suitability.get('component_scores', {})
                kecamatan_gdf.at[idx, 'temperature_score'] = component_scores.get('temperature', 0)
                kecamatan_gdf.at[idx, 'precipitation_score'] = component_scores.get('precipitation', 0)
                kecamatan_gdf.at[idx, 'humidity_score'] = component_scores.get('humidity', 0)
                
                # Add climate averages
                kecamatan_gdf.at[idx, 'avg_temperature'] = climate.get('T2M', 0)
                kecamatan_gdf.at[idx, 'avg_precipitation'] = climate.get('PRECTOTCORR', 0)
                kecamatan_gdf.at[idx, 'avg_humidity'] = climate.get('RH2M', 0)
                
                # Add risk assessment
                risk_assessment = suitability.get('risk_assessment', {})
                kecamatan_gdf.at[idx, 'overall_risk'] = risk_assessment.get('overall_risk', 'unknown')
                
            else:
                # No data available
                kecamatan_gdf.at[idx, 'suitability_score'] = 0
                kecamatan_gdf.at[idx, 'classification'] = 'No Data'
                kecamatan_gdf.at[idx, 'confidence_level'] = 'none'
        
        # Step 4: Generate GeoJSON response (same format as Next.js)
        geojson_data = kecamatan_gdf.__geo_interface__
        
        # Build request data for metadata (matching Next.js format)
        request_data = {
            "districts": districts,
            "climate_parameters": parameters,
            "analysis_period": {
                "start_year": year_start,
                "end_year": year_end
            },
            "season_filter": season,
            "aggregation_method": aggregation,
            "include_geometry": True,
            "output_format": "geojson"
        }
        
        # Create enriched response (exactly like Next.js)
        enriched_response = {
            "type": "FeatureCollection",
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "parameters_used": request_data,
                "total_districts": len(kecamatan_gdf),
                "analyzed_districts": len(analysis_results),
                "data_source": "NASA POWER 20-year dataset",
                "processing_backend": "Flask enhanced spatial analysis service",
                "analysis_method": "enhanced_rice_suitability_v2"
            },
            "features": geojson_data['features']
        }
        
        logger.info(f"Analysis complete: {len(analysis_results)} districts processed")
        
        return jsonify(enriched_response), 200
        
    except Exception as e:
        logger.error(f"Spatial analysis error: {str(e)}")
        
        return jsonify({
            "message": "Spatial analysis failed",
            "error": str(e),
            "endpoint": "/api/v1/spatial-map"
        }), 500

@spatial_api.route('/spatial-map/districts', methods=['GET'])
def get_available_districts():
    """Get available districts for analysis"""
    try:
        logger.info("Fetching available districts for analysis")
        
        geojson_loader = GeojsonLoader()
        districts = geojson_loader.get_districts_info()
        
        logger.info("Districts data retrieved successfully")
        
        return jsonify({
            "message": "Available districts retrieved successfully",
            "data": {
                "districts": districts
            },
            "total_districts": len(districts)
        }), 200
        
    except Exception as e:
        logger.error(f"Districts fetch error: {str(e)}")
        
        return jsonify({
            "message": "Failed to fetch available districts",
            "error": str(e)
        }), 500

@spatial_api.route('/spatial-map/parameters', methods=['GET'])
def get_available_climate_parameters():
    """Get available climate parameters"""
    try:
        logger.info("Fetching available climate parameters")

        # Get rice analyzer to access evaluation parameters
        rice_analyzer = RiceAnalyzer()
        evaluation_params = rice_analyzer.get_evaluation_parameters()

        # Return NASA POWER parameters
        climate_parameters = {
            "temperature": {
                "T2M": "Temperature at 2 meters (°C)",
                "T2M_MAX": "Maximum Temperature at 2 meters (°C)",
                "T2M_MIN": "Minimum Temperature at 2 meters (°C)",
            },
            "precipitation": {
                "PRECTOTCORR": "Precipitation Corrected (mm/day)",
            },
            "humidity": {
                "RH2M": "Relative Humidity at 2 meters (%)",
            },
            "solar_radiation": {
                "ALLSKY_SFC_SW_DWN": "Solar Radiation (MJ/m²/day)",
            },
            "wind": {
                "WS10M": "Wind Speed at 10 meters (m/s)",
                "WS10M_MAX": "Maximum Wind Speed at 10 meters (m/s)",
            },
        }

        return jsonify({
            "message": "Available climate parameters",
            "data": climate_parameters,
            "rice_suitability_weights": evaluation_params['weights'],
            "optimal_ranges": {
                "temperature": f"{evaluation_params['optimal_ranges']['temperature']['optimal_min']}-{evaluation_params['optimal_ranges']['temperature']['optimal_max']}°C",
                "precipitation": f"{evaluation_params['optimal_ranges']['precipitation']['optimal_min']}-{evaluation_params['optimal_ranges']['precipitation']['optimal_max']}mm/year",
                "humidity": f"{evaluation_params['optimal_ranges']['humidity']['optimal_min']}-{evaluation_params['optimal_ranges']['humidity']['optimal_max']}%",
                "solar_radiation": f"{evaluation_params['optimal_ranges']['solar']['optimal_min']}-{evaluation_params['optimal_ranges']['solar']['optimal_max']} MJ/m²/day",
                "wind": f"{evaluation_params['optimal_ranges']['wind_speed']['optimal_min']}-{evaluation_params['optimal_ranges']['wind_speed']['optimal_max']} m/s",
            },
            "quality_thresholds": evaluation_params['quality_thresholds'],
            "evaluation_methods": evaluation_params['evaluation_methods']
        }), 200
        
    except Exception as e:
        logger.error(f"Parameters fetch error: {str(e)}")
        
        return jsonify({
            "message": "Failed to fetch climate parameters",
            "error": str(e)
        }), 500