from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
from datetime import datetime
from services.geojson_loader import GeojsonLoader
from services.climate_data_service import ClimateDataService  
from services.rice_analyzer_service import RiceAnalyzer
from services.temporal_analysis_service import TemporalAnalysisService
from services.food_security_analyzer import FoodSecurityAnalyzer


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
        

@spatial_api.route('/spatial-map/food-security', methods=['GET'])
def get_food_security_analysis():
    """
    Enhanced spatial analysis with comprehensive food security metrics
    Phase 1: Climate + Temporal Stability + Resilience (without BPS)
    """
    try:
        logger.info("🍚 Starting comprehensive food security spatial analysis")
        
        # Get parameters
        districts = request.args.get('districts', 'all')
        parameters = request.args.get('parameters', 'all')
        year_start = int(request.args.get('year_start', 2004))
        year_end = int(request.args.get('year_end', 2023))
        season = request.args.get('season', 'all')
        aggregation = request.args.get('aggregation', 'mean')
        
        logger.info(f"📊 Food security analysis parameters: "
                   f"districts={districts}, year_range={year_start}-{year_end}")
        
        # Initialize services
        db = current_app.config['MONGO_DB']
        geojson_loader = GeojsonLoader()
        climate_service = ClimateDataService(db)
        rice_analyzer = RiceAnalyzer()
        food_security_analyzer = FoodSecurityAnalyzer()
        
        # Load kecamatan spatial data
        logger.info("📍 Loading kecamatan spatial data...")
        kecamatan_gdf = geojson_loader.load_kecamatan_geojson()
        
        if kecamatan_gdf is None or len(kecamatan_gdf) == 0:
            return jsonify({
                "message": "Invalid spatial data",
                "error": "Missing GeoJSON features"
            }), 500
        
        # Process each kecamatan with comprehensive food security analysis
        food_security_results = []
        
        for idx, kecamatan in kecamatan_gdf.iterrows():
            nasa_match = kecamatan['nasa_match']
            district_name = kecamatan.get('NAME_3', nasa_match)
            
            try:
                # Get collection name for this NASA location
                collection_name = climate_service.get_collection_name_for_nasa_location(nasa_match)
                
                if not collection_name:
                    logger.warning(f"No collection found for {nasa_match}")
                    continue
                
                # Load extended climate data for temporal analysis (2005-2023)
                climate_data = climate_service.load_climate_data(
                    collection_name, 2005, year_end  # Extended 19-year period
                )
                
                if climate_data is None or len(climate_data) == 0:
                    logger.warning(f"No climate data for {nasa_match}")
                    continue
                
                # Apply seasonal filter if specified
                if season != 'all':
                    climate_data = climate_service.apply_seasonal_filter(climate_data, season)
                
                # Calculate base suitability score
                aggregated_data = climate_service.aggregate_climate_data(
                    climate_data, aggregation
                )
                suitability_result = rice_analyzer.calculate_suitability_score(aggregated_data)
                
                # Convert climate_data to records format for temporal analysis
                if hasattr(climate_data, 'to_dict'):
                    climate_records = climate_data.to_dict('records')
                else:
                    climate_records = climate_data
                
                # 🍚 Comprehensive Food Security Analysis
                district_data = kecamatan.to_dict()
                fsci_analysis = food_security_analyzer.analyze_food_security(
                    district_data=district_data,
                    climate_time_series=climate_records,
                    base_suitability_score=suitability_result['score']
                )
                
                # Add comprehensive properties to kecamatan
                enhanced_properties = {
                    # Basic suitability (existing)
                    'suitability_score': suitability_result['score'],
                    'classification': suitability_result['classification'],
                    
                    # 🆕 Food Security Composite Index
                    'fsci_score': fsci_analysis.fsci_score,
                    'fsci_class': fsci_analysis.fsci_class.value,
                    
                    # 🆕 Production Capacity Index
                    'pci_score': fsci_analysis.pci.pci_score,
                    'pci_class': fsci_analysis.pci.pci_class.value,
                    'climate_suitability': fsci_analysis.pci.climate_suitability,
                    'land_quality_factor': fsci_analysis.pci.land_quality_factor,
                    'water_availability_factor': fsci_analysis.pci.water_availability_factor,
                    'risk_adjustment_factor': fsci_analysis.pci.risk_adjustment_factor,
                    
                    # 🆕 Production Stability Index
                    'psi_score': fsci_analysis.psi.psi_score,
                    'psi_class': fsci_analysis.psi.psi_class.value,
                    'temporal_stability': fsci_analysis.psi.temporal_stability,
                    'climate_variability': fsci_analysis.psi.climate_variability,
                    'trend_consistency': fsci_analysis.psi.trend_consistency,
                    'anomaly_resilience': fsci_analysis.psi.anomaly_resilience,
                    
                    # 🆕 Climate Resilience Score
                    'crs_score': fsci_analysis.crs.crs_score,
                    'crs_class': fsci_analysis.crs.crs_class.value,
                    'temperature_resilience': fsci_analysis.crs.temperature_resilience,
                    'precipitation_resilience': fsci_analysis.crs.precipitation_resilience,
                    'extreme_weather_resilience': fsci_analysis.crs.extreme_weather_resilience,
                    'adaptation_capacity': fsci_analysis.crs.adaptation_capacity,
                    
                    # 🆕 Investment Recommendation
                    'investment_recommendation': fsci_analysis.investment_recommendation,
                    
                    # Climate data
                    'avg_temperature': aggregated_data.get('T2M', 0),
                    'avg_precipitation': aggregated_data.get('PRECTOTCORR', 0),
                    'avg_humidity': aggregated_data.get('RH2M', 0),
                    
                    # Analysis metadata
                    'analysis_timestamp': fsci_analysis.analysis_timestamp,
                    'data_years': f"2005-{year_end}",
                    'temporal_analysis_enabled': True
                }
                
                # Update kecamatan properties
                for key, value in enhanced_properties.items():
                    kecamatan_gdf.at[idx, key] = value
                
                food_security_results.append(fsci_analysis)
                
                logger.info(f"🍚 Food security analysis complete for {district_name}: "
                           f"FSCI={fsci_analysis.fsci_score:.1f} "
                           f"({fsci_analysis.fsci_class.value})")
                
            except Exception as e:
                logger.error(f"Error in food security analysis for {nasa_match}: {str(e)}")
                continue
        
        # Sort results by FSCI score for priority ranking
        food_security_results.sort(key=lambda x: x.fsci_score, reverse=True)
        
        # Assign priority rankings
        for rank, analysis in enumerate(food_security_results, 1):
            analysis.priority_ranking = rank
            # Update the ranking in the GeoDataFrame
            mask = kecamatan_gdf['NAME_3'] == analysis.district_name
            if mask.any():
                kecamatan_gdf.loc[mask, 'priority_ranking'] = rank
        
        # Generate comprehensive GeoJSON response
        geojson_data = kecamatan_gdf.__geo_interface__
        
        # Calculate summary statistics
        total_districts = len(food_security_results)
        avg_fsci = np.mean([r.fsci_score for r in food_security_results]) if food_security_results else 0
        avg_pci = np.mean([r.pci.pci_score for r in food_security_results]) if food_security_results else 0
        avg_psi = np.mean([r.psi.psi_score for r in food_security_results]) if food_security_results else 0
        avg_crs = np.mean([r.crs.crs_score for r in food_security_results]) if food_security_results else 0
        
        # Classification distribution
        fsci_distribution = {}
        for result in food_security_results:
            class_name = result.fsci_class.value
            fsci_distribution[class_name] = fsci_distribution.get(class_name, 0) + 1
        
        # Top performing districts
        top_districts = [
            {
                "district": r.district_name,
                "fsci_score": r.fsci_score,
                "fsci_class": r.fsci_class.value,
                "pci_score": r.pci.pci_score,
                "psi_score": r.psi.psi_score,
                "crs_score": r.crs.crs_score,
                "investment_recommendation": r.investment_recommendation,
                "priority_ranking": r.priority_ranking
            }
            for r in food_security_results[:5]  # Top 5
        ]
        
        # Risk assessment summary
        high_risk_districts = [
            {
                "district": r.district_name,
                "fsci_score": r.fsci_score,
                "risk_factors": [
                    f"Low PCI ({r.pci.pci_score:.1f})" if r.pci.pci_score < 60 else None,
                    f"Unstable PSI ({r.psi.psi_score:.1f})" if r.psi.psi_score < 60 else None,
                    f"Low resilience CRS ({r.crs.crs_score:.1f})" if r.crs.crs_score < 60 else None
                ]
            }
            for r in food_security_results 
            if r.fsci_score < 60
        ]
        
        # Clean up risk factors (remove None values)
        for district in high_risk_districts:
            district['risk_factors'] = [rf for rf in district['risk_factors'] if rf is not None]
        
        return jsonify({
            "type": "FeatureCollection",
            "metadata": {
                "analysis_type": "comprehensive_food_security_phase1",
                "analysis_date": datetime.now().isoformat(),
                "data_period": f"2005-{year_end}",
                "total_districts": len(kecamatan_gdf),
                "analyzed_districts": total_districts,
                "data_sources": [
                    "NASA POWER Climate Data (2005-2023)",
                    "Spatial District Boundaries",
                    "Temporal Analysis Service",
                    "Food Security Analyzer"
                ],
                "analysis_components": {
                    "PCI": "Production Capacity Index (theoretical)",
                    "PSI": "Production Stability Index (temporal analysis)", 
                    "CRS": "Climate Resilience Score",
                    "FSCI": "Food Security Composite Index"
                },
                "summary_statistics": {
                    "average_scores": {
                        "fsci_score": round(avg_fsci, 2),
                        "pci_score": round(avg_pci, 2),
                        "psi_score": round(avg_psi, 2),
                        "crs_score": round(avg_crs, 2)
                    },
                    "fsci_distribution": fsci_distribution,
                    "high_risk_districts_count": len(high_risk_districts),
                    "temporal_analysis_coverage": "100%"
                },
                "top_performing_districts": top_districts,
                "high_risk_districts": high_risk_districts[:3],  # Top 3 highest risk
                "parameters_used": {
                    "year_range": f"{year_start}-{year_end}",
                    "temporal_range": f"2005-{year_end}",
                    "season": season,
                    "aggregation": aggregation,
                    "fsci_weights": {
                        "PCI": "45%",
                        "PSI": "30%", 
                        "CRS": "25%"
                    }
                },
                "investment_insights": {
                    "prime_areas": len([r for r in food_security_results if r.fsci_score >= 80]),
                    "development_areas": len([r for r in food_security_results if 60 <= r.fsci_score < 80]),
                    "risk_areas": len([r for r in food_security_results if r.fsci_score < 60]),
                    "recommendation": "Focus investment on development areas with high PCI but low PSI/CRS"
                }
            },
            "features": geojson_data['features']
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Comprehensive food security analysis error: {str(e)}")
        
        return jsonify({
            "message": "Comprehensive food security analysis failed",
            "error": str(e),
            "endpoint": "/api/v1/spatial-map/food-security"
        }), 500