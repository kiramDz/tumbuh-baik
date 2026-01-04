import logging
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from services.climate_data_service import ClimateDataService
from services.spatial_analysis import create_spatial_connector

logger = logging.getLogger(__name__)

# Create blueprint
nasa_api_bp = Blueprint('nasa_api', __name__, url_prefix='/api/v1/nasa')


@nasa_api_bp.route('/climate', methods=['GET'])
def get_nasa_climate_data():
    """Get NASA POWER climate data as GeoJSON"""
    try:
        logger.info("Getting NASA POWER climate data")
        
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Create spatial connector
        spatial_connector = create_spatial_connector(db)
        
        # Get available NASA datasets
        datasets = spatial_connector.get_nasa_datasets()
        
        if not datasets:
            return jsonify({'error': 'No NASA POWER datasets available'}), 404
        
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
                    "parameters": dataset.parameters
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [dataset.longitude, dataset.latitude]
                }
            }
            features.append(feature)
        
        geojson_response = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return jsonify(geojson_response), 200
        
    except Exception as e:
        logger.error(f"Error getting NASA climate data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@nasa_api_bp.route('/climate/districts', methods=['GET'])
def get_nasa_districts():
    """Get available districts/locations from NASA POWER datasets"""
    try:
        logger.info("Getting available districts from NASA datasets")
        
        # Get db from Flask app context
        db = current_app.config['MONGO_DB']
        
        # Create spatial connector
        spatial_connector = create_spatial_connector(db)
        
        # Get datasets
        datasets = spatial_connector.get_nasa_datasets()
        
        districts = []
        for dataset in datasets:
            # Extract district name from dataset name
            district_name = dataset.name.replace("Nasa", "").replace("Data NASA", "").strip()
            
            districts.append({
                "name": district_name,
                "collection_name": dataset.collection_name,
                "coordinates": {
                    "lat": dataset.latitude,
                    "lng": dataset.longitude
                },
                "date_range": {
                    "start": dataset.date_range[0].isoformat(),
                    "end": dataset.date_range[1].isoformat()
                }
            })
        
        return jsonify({
            "districts": districts,
            "total": len(districts)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting districts: {str(e)}")
        return jsonify({'error': str(e)}), 500