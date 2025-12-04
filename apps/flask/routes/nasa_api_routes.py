import logging
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from services.climate_data_service import ClimateDataService
from services.spatial_analysis import create_spatial_connector

logger = logging.getLogger(__name__)

# Create blueprint with uniform naming
nasa_api_bp = Blueprint('nasa_api', __name__, url_prefix='/api/v1/nasa')


@nasa_api_bp.route('/climate', methods=['GET'])
def get_nasa_climate_datasets():
    """Get all available NASA POWER datasets for spatial analysis"""
    try:
        logger.info("🗺️ Getting NASA POWER datasets for spatial analysis")
        
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Create spatial connector
        spatial_connector = create_spatial_connector(db)
        
        # Get datasets summary
        summary = spatial_connector.get_datasets_summary()
        
        return jsonify({
            "status": "success",
            "message": "NASA POWER datasets retrieved successfully",
            "data": summary,
            "fetch_timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting NASA datasets: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to get NASA datasets",
            "error": str(e)
        }), 500


def _perform_nasa_analysis(request_data):
    """Common NASA analysis logic for both GET and POST"""
    try:
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Extract parameters
        districts = request_data.get('districts', 'all')
        climate_parameters = request_data.get('climate_parameters', 'all')
        analysis_period = request_data.get('analysis_period', {})
        season_filter = request_data.get('season_filter', 'all')
        aggregation_method = request_data.get('aggregation_method', 'mean')
        
        logger.info(f"📊 Analysis parameters: districts={districts}, parameters={climate_parameters}")
        
        # Create spatial connector
        spatial_connector = create_spatial_connector(db)
        
        # Get available NASA datasets
        datasets = spatial_connector.get_nasa_datasets()
        
        if not datasets:
            return jsonify({
                "status": "error",
                "message": "No NASA POWER datasets available for analysis",
                "error": "No data found"
            }), 404
        
        # Create GeoJSON features from datasets
        features = []
        for dataset in datasets:
            feature = {
                "type": "Feature",
                "properties": {
                    "name": dataset.name,
                    "collection_name": dataset.collection_name,
                    "total_records": dataset.total_records,
                    "date_range": {
                        "start": dataset.date_range[0].isoformat(),
                        "end": dataset.date_range[1].isoformat()
                    },
                    "parameters": dataset.parameters,
                    "suitability_score": 0.0,  # Placeholder - will be calculated
                    "final_score": 0.0,  # Placeholder - will be calculated
                    "classification": "Pending Analysis"  # Placeholder
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [dataset.longitude, dataset.latitude]
                }
            }
            features.append(feature)
        
        geojson_response = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "analysis_type": "nasa_climate_potential",
                "total_locations": len(features),
                "analysis_parameters": {
                    "districts": districts,
                    "climate_parameters": climate_parameters,
                    "analysis_period": analysis_period,
                    "season_filter": season_filter,
                    "aggregation_method": aggregation_method
                },
                "fetch_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ NASA climate analysis completed: {len(features)} locations")
        
        return jsonify({
            "status": "success", 
            **geojson_response
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error in NASA analysis helper: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "NASA climate analysis failed",
            "error": str(e)
        }), 500

@nasa_api_bp.route('/climate/analysis', methods=['POST'])
def nasa_climate_analysis():
    """NASA climate potential spatial analysis endpoint"""
    try:
        logger.info("🗺️ Starting NASA climate potential analysis (POST)")
        
        # Get request data with better error handling
        try:
            request_data = request.get_json(force=True) or {}
        except Exception:
            # If no JSON provided, use defaults
            request_data = {}
            
        # Call the common analysis function
        return _perform_nasa_analysis(request_data)
        
    except Exception as e:
        logger.error(f"❌ Error in NASA climate analysis (POST): {str(e)}")
        return jsonify({
            "status": "error",
            "message": "NASA climate analysis failed",
            "error": str(e)
        }), 500

@nasa_api_bp.route('/climate/districts', methods=['GET'])
def get_nasa_districts():
    """Get available districts/locations from NASA POWER datasets"""
    try:
        logger.info("📍 Getting available districts from NASA datasets")
        
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Create spatial connector
        spatial_connector = create_spatial_connector(db)
        
        # Get datasets
        datasets = spatial_connector.get_nasa_datasets()
        
        districts = []
        for i, dataset in enumerate(datasets):
            # Extract district name from dataset name
            district_name = dataset.name.replace("Nasa", "").replace("Data NASA", "").strip()
            
            districts.append({
                "id": f"district_{i+1}",
                "name": district_name,
                "collection_name": dataset.collection_name,
                "coordinates": {
                    "latitude": dataset.latitude,
                    "longitude": dataset.longitude
                },
                "data_availability": {
                    "start_date": dataset.date_range[0].isoformat(),
                    "end_date": dataset.date_range[1].isoformat(),
                    "total_records": dataset.total_records
                }
            })
        
        return jsonify({
            "status": "success",
            "message": "Available districts retrieved successfully",
            "districts": districts,
            "total": len(districts),
            "fetch_timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting districts: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to get available districts",
            "error": str(e)
        }), 500

@nasa_api_bp.route('/climate/datasets/<collection_name>', methods=['GET'])
def get_nasa_dataset_details(collection_name: str):
    """Get detailed information about specific NASA dataset"""
    try:
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Initialize climate data service
        climate_service = ClimateDataService(db)
        
        # Get available parameters
        parameters = climate_service.get_available_parameters(collection_name)
        
        if not parameters:
            return jsonify({
                "status": "error",
                "message": f"Dataset {collection_name} not found or has no parameters"
            }), 404
        
        # Get sample data for statistics
        sample_df = climate_service.load_climate_data(collection_name, 2020, 2024)
        
        if sample_df is not None:
            statistics = climate_service.get_climate_statistics(sample_df)
            quality_report = climate_service.validate_data_quality(sample_df)
        else:
            statistics = {}
            quality_report = {}
        
        return jsonify({
            "status": "success",
            "collection_name": collection_name,
            "available_parameters": parameters,
            "statistics": statistics,
            "quality_report": quality_report,
            "fetch_timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting NASA dataset details: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Error handlers
@nasa_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/v1/nasa/climate',
            '/api/v1/nasa/climate/analysis',
            '/api/v1/nasa/climate/districts',
            '/api/v1/nasa/climate/datasets/<collection_name>'
        ]
    }), 404

@nasa_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error': 'Internal server error',
        'message': 'Please check server logs for details'
    }), 500


