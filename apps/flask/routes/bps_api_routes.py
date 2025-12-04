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
        
        logger.info(f"Historical request: {kabupaten_name} ({start_year}-{end_year})")
        
        # Fetch historical data
        historical_data = bps_service.fetch_kabupaten_historical_data(
            kabupaten_name, start_year, end_year
        )
        
        if not historical_data:
            return jsonify({
                'status': 'error',
                'error': f'No historical data found for {kabupaten_name}',
                'available_kabupaten': bps_service.target_kabupaten
            }), 404
        
        # Format simple response
        multi_year_production = {}
        for year, record in historical_data.items():
            multi_year_production[str(year)] = {
                "produksi_padi_ton": record.produksi_padi_ton,
                "produksi_beras_ton": record.produksi_beras_ton,
                "conversion_rate_percent": round((record.produksi_beras_ton / record.produksi_padi_ton) * 100, 2) if record.produksi_padi_ton > 0 else 0
            }
        
        response_data = {
            'status': 'success',
            'kabupaten': kabupaten_name,
            'kode_wilayah': list(historical_data.values())[0].kode_wilayah,
            'multi_year_data': multi_year_production,
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in historical production endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

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
        
        logger.info(f"Multi-year summary request: {start_year}-{end_year}")
        
        # Fetch multi-year data
        multi_year_data = bps_service.fetch_multi_year_production_data(start_year, end_year)
        
        if not multi_year_data:
            return jsonify({
                'status': 'error',
                'error': 'No multi-year data available'
            }), 404
        
        # Process data by kabupaten - simple format
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
                yearly_data = {}
                for year, record in kabupaten_historical.items():
                    yearly_data[str(year)] = {
                        "produksi_padi_ton": record.produksi_padi_ton,
                        "produksi_beras_ton": record.produksi_beras_ton,
                        "conversion_rate_percent": round((record.produksi_beras_ton / record.produksi_padi_ton) * 100, 2) if record.produksi_padi_ton > 0 else 0
                    }
                
                # Simple kabupaten summary
                kabupaten_summaries[kabupaten_name] = {
                    "kode_wilayah": list(kabupaten_historical.values())[0].kode_wilayah,
                    "years_with_data": sorted(kabupaten_historical.keys()),
                    "data_coverage": f"{len(kabupaten_historical)}/{end_year - start_year + 1} years",
                    "yearly_production": yearly_data,
                    "latest_production": {
                        "year": max(kabupaten_historical.keys()),
                        "produksi_padi_ton": kabupaten_historical[max(kabupaten_historical.keys())].produksi_padi_ton,
                        "produksi_beras_ton": kabupaten_historical[max(kabupaten_historical.keys())].produksi_beras_ton
                    }
                }
        
        return jsonify({
            'status': 'success',
            'analysis_period': f"{start_year}-{end_year}",
            'total_years_requested': end_year - start_year + 1,
            'years_with_data': sorted(multi_year_data.keys()),
            'data_completeness': f"{len(multi_year_data)}/{end_year - start_year + 1} years",
            'kabupaten_analysis': kabupaten_summaries,
            'fetch_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in multi-year summary endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@bps_api_bp.route('/rice-production/available-years', methods=['GET'])
def get_available_years():
    """
    Get available years for BPS data
    """
    return jsonify({
        'status': 'success',
        'available_years': [2024, 2023, 2022, 2021, 2020, 2019, 2018],
        'recommended_range': {
            'start_year': 2018,
            'end_year': 2024,
            'total_years': 7
        },
        'note': 'Year availability depends on BPS API data publication schedule'
    })

@bps_api_bp.route('/rice-production/target-kabupaten', methods=['GET'])
def get_target_kabupaten():
    """
    Get list of target kabupaten for analysis
    """
    return jsonify({
        'status': 'success',
        'target_kabupaten': bps_service.target_kabupaten,
        'total_kabupaten': len(bps_service.target_kabupaten),
        'kabupaten_mapping': bps_service.kabupaten_mapping
    })

# Error handlers
@bps_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/v1/bps/rice-production/kabupaten/<name>/historical',
            '/api/v1/bps/rice-production/multi-year-summary',
            '/api/v1/bps/rice-production/available-years',
            '/api/v1/bps/rice-production/target-kabupaten'
        ]
    }), 404

@bps_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error': 'Internal server error',
        'message': 'Please check server logs for details'
    }), 500