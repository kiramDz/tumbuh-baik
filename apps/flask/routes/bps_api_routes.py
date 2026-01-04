import logging
from datetime import datetime
from flask import Blueprint, jsonify, request
from services.bps_api_service import BPSApiService

logger = logging.getLogger(__name__)

# Create blueprint
bps_api_bp = Blueprint('bps_api', __name__, url_prefix='/api/v1/bps')

# Initialize BPS service
bps_service = BPSApiService()

@bps_api_bp.route('/rice-production/kabupaten/<kabupaten_name>/historical', methods=['GET'])
def get_kabupaten_historical_production(kabupaten_name: str):
    """
    Get multi-year historical production data for specific kabupaten
    
    Query Parameters:
    - start_year: Starting year (default: 2018)
    - end_year: Ending year (default: 2024)
    """
    try:
        start_year = int(request.args.get('start_year', 2018))
        end_year = int(request.args.get('end_year', 2024))
        
        logger.info(f"Fetching historical data for {kabupaten_name} ({start_year}-{end_year})")
        
        # Fetch historical data
        historical_data = bps_service.fetch_kabupaten_historical_data(
            kabupaten_name, start_year, end_year
        )
        
        if not historical_data:
            return jsonify({
                'error': f'No historical data found for {kabupaten_name}'
            }), 404
        
        # Format simple response
        padi_ton = {}
        for year, record in historical_data.items():
            padi_ton[str(year)] = record.padi_ton

        response_data = {
            'kabupaten': kabupaten_name,
            'padi_ton': padi_ton
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in historical production endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bps_api_bp.route('/rice-production/multi-year-summary', methods=['GET'])
def get_multi_year_production_summary():
    """
    Get multi-year production summary for all target kabupaten
    
    Query Parameters:
    - start_year: Starting year (default: 2018)  
    - end_year: Ending year (default: 2024)
    """
    try:
        start_year = int(request.args.get('start_year', 2018))
        end_year = int(request.args.get('end_year', 2024))
        
        logger.info(f"Fetching multi-year summary ({start_year}-{end_year})")
        
        # Fetch multi-year data
        multi_year_data = bps_service.fetch_multi_year_production_data(start_year, end_year)
        
        if not multi_year_data:
            return jsonify({'error': 'No multi-year data available'}), 404
        
        # Process data by kabupaten
        kabupaten_summaries = {}
        
        for kabupaten_name in bps_service.target_kabupaten:
            # Extract historical data for this kabupaten
            kabupaten_historical = {}
            for year, year_records in multi_year_data.items():
                for record in year_records:
                    if record.kabupaten == kabupaten_name:
                        kabupaten_historical[year] = record
                        break
            
            if kabupaten_historical:
                # Format yearly data
                padi_ton = {}
                for year, record in kabupaten_historical.items():
                    padi_ton[str(year)] = record.padi_ton
                
                kabupaten_summaries[kabupaten_name] = {
                    'padi_ton': padi_ton
                }
        
        return jsonify({
            'analysis_period': f"{start_year}-{end_year}",
            'kabupaten_data': kabupaten_summaries
        })
        
    except Exception as e:
        logger.error(f"Error in multi-year summary endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500