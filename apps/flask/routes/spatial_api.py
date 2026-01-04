from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
from datetime import datetime
from services.geojson_loader import GeojsonLoader
from services.climate_data_service import ClimateDataService  
from services.rice_analyzer_service import RiceAnalyzer
from services.food_security_analyzer import FoodSecurityAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint with /api prefix to match Next.js structure
spatial_api = Blueprint('spatial_api', __name__, url_prefix='/api/v1')

@spatial_api.route('/spatial-map', methods=['GET'])
def get_climate_potential_analysis():
    """
    Main endpoint for climate potential analysis with simplified FSI
    """
    try:
        logger.info("Starting FSI spatial analysis request")

        # Get query parameters (simplified - removed season)
        districts = request.args.get('districts', 'all')
        year_start = int(request.args.get('year_start', 2018))
        year_end = int(request.args.get('year_end', 2023))
        aggregation = request.args.get('aggregation', 'mean')

        logger.info(f"FSI analysis parameters: districts={districts}, "
                   f"year_range={year_start}-{year_end}, aggregation={aggregation}")

        # Initialize services
        db = current_app.config['MONGO_DB']
        geojson_loader = GeojsonLoader()
        climate_service = ClimateDataService(db)
        rice_analyzer = RiceAnalyzer()
        food_security_analyzer = FoodSecurityAnalyzer()
        
        # Step 1: Load kecamatan spatial data
        logger.info("Loading kecamatan spatial data...")
        kecamatan_gdf = geojson_loader.load_kecamatan_geojson()
        
        if kecamatan_gdf is None or len(kecamatan_gdf) == 0:
            return jsonify({
                "message": "Invalid spatial analysis response",
                "error": "Missing GeoJSON features"
            }), 500
        
        # Step 2: Analyze each kecamatan with FSI
        analysis_results = {}
        fsi_results = []
        
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
                                
                # Aggregate climate data using specified method
                aggregated_data = climate_service.aggregate_climate_data(
                    climate_data, aggregation
                )
                
                # Calculate rice suitability score
                suitability_result = rice_analyzer.calculate_suitability_score(aggregated_data)
                
                # Convert climate_data to records format for FSI analysis
                if hasattr(climate_data, 'to_dict'):
                    climate_records = climate_data.to_dict('records')
                else:
                    climate_records = climate_data if isinstance(climate_data, list) else []
                
                # 🍚 Food Security Index Analysis (Simplified)
                district_data = kecamatan.to_dict()
                fsi_analysis = food_security_analyzer.analyze_food_security(
                    district_data=district_data,
                    climate_time_series=climate_records,
                    base_suitability_score=suitability_result['score']
                )
                
                analysis_results[nasa_match] = {
                    'climate_data': aggregated_data,
                    'suitability': suitability_result,
                    'fsi_analysis': fsi_analysis
                }
                
                fsi_results.append(fsi_analysis)
                
                logger.info(f"Analyzed {nasa_match}: FSI={fsi_analysis.fsi_score:.1f} "
                           f"({fsi_analysis.fsi_class.value})")
                
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
                fsi_analysis = result['fsi_analysis']
                
                # Add FSI properties
                kecamatan_gdf.at[idx, 'fsi_score'] = fsi_analysis.fsi_score
                kecamatan_gdf.at[idx, 'fsi_class'] = fsi_analysis.fsi_class.value
                kecamatan_gdf.at[idx, 'natural_resources_score'] = fsi_analysis.natural_resources_score
                kecamatan_gdf.at[idx, 'availability_score'] = fsi_analysis.availability_score
                
                # Add suitability properties (for backward compatibility)
                kecamatan_gdf.at[idx, 'suitability_score'] = suitability['score']
                kecamatan_gdf.at[idx, 'classification'] = suitability['classification']
                kecamatan_gdf.at[idx, 'confidence_level'] = suitability.get('confidence_level', 'medium')
                
                # Add component scores
                component_scores = suitability.get('component_scores', {})
                kecamatan_gdf.at[idx, 'temperature_score'] = component_scores.get('temperature', 0)
                kecamatan_gdf.at[idx, 'precipitation_score'] = component_scores.get('precipitation', 0)
                kecamatan_gdf.at[idx, 'humidity_score'] = component_scores.get('humidity', 0)
                
                # Add climate averages with fallbacks
                kecamatan_gdf.at[idx, 'avg_temperature'] = climate.get('T2M', 26.0)
                kecamatan_gdf.at[idx, 'avg_precipitation'] = climate.get('PRECTOTCORR', 8.0)
                kecamatan_gdf.at[idx, 'avg_humidity'] = climate.get('RH2M', 75.0)
                
                # Add risk assessment
                risk_assessment = suitability.get('risk_assessment', {})
                kecamatan_gdf.at[idx, 'overall_risk'] = risk_assessment.get('overall_risk', 'moderate')
                
                # Add analysis metadata
                kecamatan_gdf.at[idx, 'analysis_timestamp'] = fsi_analysis.analysis_timestamp
                
            else:
                # No data available - set defaults
                kecamatan_gdf.at[idx, 'fsi_score'] = 0
                kecamatan_gdf.at[idx, 'fsi_class'] = 'No Data'
                kecamatan_gdf.at[idx, 'natural_resources_score'] = 0
                kecamatan_gdf.at[idx, 'availability_score'] = 0
                kecamatan_gdf.at[idx, 'suitability_score'] = 0
                kecamatan_gdf.at[idx, 'classification'] = 'No Data'
                kecamatan_gdf.at[idx, 'confidence_level'] = 'none'
                kecamatan_gdf.at[idx, 'avg_temperature'] = 0
                kecamatan_gdf.at[idx, 'avg_precipitation'] = 0
                kecamatan_gdf.at[idx, 'avg_humidity'] = 0
                kecamatan_gdf.at[idx, 'overall_risk'] = 'unknown'
        
        # Step 4: Generate GeoJSON response
        geojson_data = kecamatan_gdf.__geo_interface__
        
        # Calculate FSI summary statistics
        total_districts = len(fsi_results)
        avg_fsi = np.mean([r.fsi_score for r in fsi_results]) if fsi_results else 0
        avg_natural_resources = np.mean([r.natural_resources_score for r in fsi_results]) if fsi_results else 0
        avg_availability = np.mean([r.availability_score for r in fsi_results]) if fsi_results else 0
        
        # FSI classification distribution
        fsi_distribution = {}
        for result in fsi_results:
            class_name = result.fsi_class.value
            fsi_distribution[class_name] = fsi_distribution.get(class_name, 0) + 1
        
        # Sort by FSI score for ranking
        fsi_results.sort(key=lambda x: x.fsi_score, reverse=True)
        
        # Top performing districts
        top_districts = [
            {
                "district": r.district_name,
                "fsi_score": r.fsi_score,
                "fsi_class": r.fsi_class.value,
                "natural_resources_score": r.natural_resources_score,
                "availability_score": r.availability_score,
                "ranking": idx + 1
            }
            for idx, r in enumerate(fsi_results[:5])  # Top 5
        ]
        
        request_data = {
            "districts": districts,
            "analysis_period": {
                "start_year": year_start,
                "end_year": year_end
            },
            "aggregation_method": aggregation,
            "include_geometry": True,
            "output_format": "geojson"
        }
        
        # Create simplified response
        simplified_response = {
            "type": "FeatureCollection",
            "metadata": {
                "analysis_type": "food_security_index_spatial_analysis",
                "analysis_date": datetime.now().isoformat(),
                "parameters_used": request_data,
                "total_districts": len(kecamatan_gdf),
                "analyzed_districts": total_districts,
                "data_source": "NASA POWER climate dataset",
                "analysis_method": "simplified_food_security_index_v2",
                "fsi_components": {
                    "natural_resources": "Climate sustainability and resilience (60%)",
                    "availability": "Food supply adequacy proxy (40%)"
                },
                "summary_statistics": {
                    "average_scores": {
                        "fsi_score": round(avg_fsi, 2),
                        "natural_resources_score": round(avg_natural_resources, 2),
                        "availability_score": round(avg_availability, 2)
                    },
                    "fsi_distribution": fsi_distribution
                },
                "top_performing_districts": top_districts,
                "data_period": f"{year_start}-{year_end}",
                "temporal_coverage": f"{year_end - year_start + 1} years"
            },
            "features": geojson_data['features']
        }
        
        logger.info(f"FSI spatial analysis complete: {total_districts} districts processed, "
                   f"average FSI: {avg_fsi:.1f}")
        
        if len(fsi_results) >= 5:
            logger.info("Applying hybrid FSI classification (percentile + BPS validation)...")
            fsi_results = food_security_analyzer.apply_hybrid_fsi_classification(fsi_results)
            
            # Update GeoDataFrame with new classifications
            for idx, kecamatan in kecamatan_gdf.iterrows():
                nasa_match = kecamatan['nasa_match']
                
                if nasa_match in analysis_results:
                    for fsi_result in fsi_results:
                        if fsi_result.district_name.replace(' ', '') == nasa_match.replace(' ', ''):
                            kecamatan_gdf.at[idx, 'fsi_class'] = fsi_result.fsi_class.value
                            break
        else:
            logger.info("Using fixed thresholds (insufficient data for hybrid classification)")
        
        return jsonify(simplified_response), 200
        
    except Exception as e:
        logger.error(f"FSI spatial analysis error: {str(e)}")
        
        return jsonify({
            "message": "FSI spatial analysis failed",
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