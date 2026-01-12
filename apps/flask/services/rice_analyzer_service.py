import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

logger = logging.getLogger(__name__)

class RiceAnalyzer:
    """Enhanced rice suitability analysis with NASA preprocessing-style evaluation"""
    
    def __init__(self):
        # Optimal ranges for rice cultivation (based on agricultural research)
        self.optimal_ranges = {
            'temperature': {
                'min': 18, 'max': 40, 
                'optimal_min': 22, 'optimal_max': 32,
                'critical_min': 15, 'critical_max': 42  # Beyond these = crop failure
            },
            'precipitation': {
                'min': 800, 'max': 3000, 
                'optimal_min': 1200, 'optimal_max': 2000,
                'critical_min': 600, 'critical_max': 3500
            },
            'humidity': {
                'min': 50, 'max': 95, 
                'optimal_min': 70, 'optimal_max': 85,
                'critical_min': 40, 'critical_max': 98
            },
            'solar': {
                'min': 8, 'max': 35, 
                'optimal_min': 15, 'optimal_max': 25,
                'critical_min': 5, 'critical_max': 40
            },
            'wind_speed': {
                'min': 0.2, 'max': 8.0, 
                'optimal_min': 1.0, 'optimal_max': 4.0,
                'critical_min': 0, 'critical_max': 12.0
            }
        }
        
        # Weights for rice suitability (based on agricultural importance)
        self.weights = {
            'temperature': 0.25,      # Critical for growth phases
            'precipitation': 0.35,    # Most important - water requirement
            'humidity': 0.20,         # Important for disease prevention
            'solar': 0.15,           # Photosynthesis requirement
            'wind_speed': 0.05       # Minor but affects pollination
        }
        
        # Quality thresholds (similar to NASA preprocessing R² thresholds)
        self.quality_thresholds = {
            'excellent': 85,
            'good': 70,
            'fair': 55,
            'marginal': 40,
            'poor': 0
        }
        
    def calculate_suitability_score(self, climate_data: Dict[str, float]) -> Dict[str, Any]:
        """Enhanced suitability calculation with quality validation"""
        try:
            logger.info("Calculating rice suitability with enhanced evaluation")
            
            # Step 1: Validate input data quality
            data_quality = self._validate_input_data(climate_data)
            
            if data_quality['quality_score'] < 50:
                logger.warning(f"Low input data quality: {data_quality['quality_score']:.1f}%")
            
            # Step 2: Calculate component scores with detailed evaluation
            component_analysis = self._calculate_component_scores(climate_data)
            
            # Step 3: Apply weighted scoring with confidence intervals
            weighted_score = self._calculate_weighted_score(component_analysis['scores'])
            
            # Step 4: Apply quality adjustments
            adjusted_score = self._apply_quality_adjustments(
                weighted_score, 
                data_quality, 
                component_analysis
            )
            
            # Step 5: Generate detailed classification
            classification_result = self._classify_suitability_enhanced(adjusted_score)
            
            # Step 6: Calculate risk assessment
            risk_assessment = self._assess_cultivation_risks(climate_data, component_analysis)
            
            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                component_analysis, 
                classification_result,
                risk_assessment
            )
            
            result = {
                'score': round(adjusted_score, 2),
                'classification': classification_result['class'],
                'confidence_level': classification_result['confidence'],
                'component_scores': component_analysis['scores'],
                'component_analysis': component_analysis['analysis'],
                'data_quality': data_quality,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'climate_values': self._extract_climate_values(climate_data),
                'weights_used': self.weights,
                'analysis_metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'method': 'enhanced_rice_suitability_v2',
                    'quality_adjusted': True
                }
            }
            
            logger.info(f"Suitability calculated: {adjusted_score:.1f} ({classification_result['class']}) "
                       f"- Confidence: {classification_result['confidence']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in suitability calculation: {str(e)}")
            return self._create_error_result(str(e))
    
    def _validate_input_data(self, climate_data: Dict[str, float]) -> Dict[str, Any]:
        """Validate input data quality (inspired by NASA preprocessing validation)"""
        
        required_params = ['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M']
        
        validation_result = {
            'quality_score': 100.0,
            'missing_parameters': [],
            'out_of_range_parameters': [],
            'warnings': [],
            'data_completeness': 0.0
        }
        
        # Check for missing parameters
        missing_params = [param for param in required_params if param not in climate_data or climate_data[param] is None]
        validation_result['missing_parameters'] = missing_params
        
        # Calculate completeness
        available_params = len(required_params) - len(missing_params)
        validation_result['data_completeness'] = (available_params / len(required_params)) * 100
        
        # Quality penalty for missing data
        if missing_params:
            completeness_penalty = len(missing_params) * 15  # 15% penalty per missing param
            validation_result['quality_score'] -= completeness_penalty
            validation_result['warnings'].append(f"Missing parameters: {missing_params}")
        
        # Check for out-of-range values
        for param, value in climate_data.items():
            if param in ['T2M'] and value is not None:
                if not (-50 <= value <= 60):  # Reasonable temperature range
                    validation_result['out_of_range_parameters'].append(f"{param}: {value}")
                    validation_result['quality_score'] -= 10
                    
            elif param in ['PRECTOTCORR'] and value is not None:
                if not (0 <= value <= 1000):  # Reasonable precipitation range (mm/day)
                    validation_result['out_of_range_parameters'].append(f"{param}: {value}")
                    validation_result['quality_score'] -= 10
                    
            elif param in ['RH2M'] and value is not None:
                if not (0 <= value <= 100):  # Humidity percentage
                    validation_result['out_of_range_parameters'].append(f"{param}: {value}")
                    validation_result['quality_score'] -= 10
        
        validation_result['quality_score'] = max(0, validation_result['quality_score'])
        
        return validation_result
    
    def _calculate_component_scores(self, climate_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate detailed component scores with analysis"""
        
        component_scores = {}
        component_analysis = {}
        
        # Temperature analysis
        temperature = climate_data.get('T2M', 0)
        temp_result = self._evaluate_temperature_enhanced(temperature)
        component_scores['temperature'] = temp_result['score']
        component_analysis['temperature'] = temp_result['analysis']
        
        # Precipitation analysis (convert mm/day to mm/year)
        precipitation_daily = climate_data.get('PRECTOTCORR', 0)
        precipitation_annual = precipitation_daily * 365
        precip_result = self._evaluate_precipitation_enhanced(precipitation_annual)
        component_scores['precipitation'] = precip_result['score']
        component_analysis['precipitation'] = precip_result['analysis']
        
        # Humidity analysis
        humidity = climate_data.get('RH2M', 0)
        humidity_result = self._evaluate_humidity_enhanced(humidity)
        component_scores['humidity'] = humidity_result['score']
        component_analysis['humidity'] = humidity_result['analysis']
        
        # Solar radiation analysis
        solar = climate_data.get('ALLSKY_SFC_SW_DWN', 0)
        solar_result = self._evaluate_solar_enhanced(solar)
        component_scores['solar'] = solar_result['score']
        component_analysis['solar'] = solar_result['analysis']
        
        # Wind speed analysis
        wind_speed = climate_data.get('WS10M', 0)
        wind_result = self._evaluate_wind_enhanced(wind_speed)
        component_scores['wind_speed'] = wind_result['score']
        component_analysis['wind_speed'] = wind_result['analysis']
        
        return {
            'scores': component_scores,
            'analysis': component_analysis
        }
    
    def _evaluate_temperature_enhanced(self, temperature: float) -> Dict[str, Any]:
        """Enhanced temperature evaluation with detailed analysis"""
        ranges = self.optimal_ranges['temperature']
        
        # Calculate score using multiple methods
        score_linear = self._calculate_linear_score(temperature, ranges)
        score_gaussian = self._calculate_gaussian_score(temperature, ranges)
        
        # Weighted combination (70% linear, 30% gaussian for smoothness)
        final_score = 0.7 * score_linear + 0.3 * score_gaussian
        
        # Determine status and risks
        if temperature < ranges['critical_min'] or temperature > ranges['critical_max']:
            status = "critical"
            risk_level = "high"
        elif ranges['optimal_min'] <= temperature <= ranges['optimal_max']:
            status = "optimal"
            risk_level = "low"
        elif ranges['min'] <= temperature <= ranges['max']:
            status = "suitable"
            risk_level = "medium"
        else:
            status = "marginal"
            risk_level = "high"
        
        analysis = {
            'value': temperature,
            'status': status,
            'risk_level': risk_level,
            'optimal_range': f"{ranges['optimal_min']}-{ranges['optimal_max']}°C",
            'deviation_from_optimal': abs(temperature - (ranges['optimal_min'] + ranges['optimal_max'])/2),
            'score_breakdown': {
                'linear': round(score_linear, 2),
                'gaussian': round(score_gaussian, 2)
            }
        }
        
        return {
            'score': round(final_score, 2),
            'analysis': analysis
        }
    
    def _evaluate_precipitation_enhanced(self, precipitation: float) -> Dict[str, Any]:
        """Enhanced precipitation evaluation"""
        ranges = self.optimal_ranges['precipitation']
        
        # Special handling for precipitation (asymmetric penalty)
        if precipitation < ranges['min']:
            # Severe penalty for drought conditions
            score = max(0, 60 * (precipitation / ranges['min']))
            status = "drought_risk"
            risk_level = "very_high"
        elif precipitation > ranges['max']:
            # Moderate penalty for excess water
            excess_ratio = (precipitation - ranges['max']) / ranges['max']
            score = max(20, 80 - (excess_ratio * 30))
            status = "flood_risk"
            risk_level = "high"
        elif ranges['optimal_min'] <= precipitation <= ranges['optimal_max']:
            score = 100.0
            status = "optimal"
            risk_level = "low"
        else:
            # Linear interpolation for suitable range
            if precipitation < ranges['optimal_min']:
                score = 60 + 40 * (precipitation - ranges['min']) / (ranges['optimal_min'] - ranges['min'])
                status = "below_optimal"
            else:
                score = 60 + 40 * (ranges['max'] - precipitation) / (ranges['max'] - ranges['optimal_max'])
                status = "above_optimal"
            risk_level = "medium"
        
        # Water stress index
        optimal_center = (ranges['optimal_min'] + ranges['optimal_max']) / 2
        water_stress = abs(precipitation - optimal_center) / optimal_center
        
        analysis = {
            'value': precipitation,
            'daily_average': round(precipitation / 365, 2),
            'status': status,
            'risk_level': risk_level,
            'water_stress_index': round(water_stress, 3),
            'optimal_range': f"{ranges['optimal_min']}-{ranges['optimal_max']} mm/year"
        }
        
        return {
            'score': round(score, 2),
            'analysis': analysis
        }
    
    def _evaluate_humidity_enhanced(self, humidity: float) -> Dict[str, Any]:
        """Enhanced humidity evaluation with disease risk assessment"""
        ranges = self.optimal_ranges['humidity']
        
        score = self._calculate_linear_score(humidity, ranges)
        
        # Disease risk assessment
        if humidity > 90:
            disease_risk = "very_high"
            risk_factors = ["fungal_diseases", "bacterial_blight"]
        elif humidity > 85:
            disease_risk = "high"
            risk_factors = ["brown_spot", "blast"]
        elif humidity < 60:
            disease_risk = "medium"
            risk_factors = ["drought_stress", "poor_pollination"]
        else:
            disease_risk = "low"
            risk_factors = []
        
        analysis = {
            'value': humidity,
            'disease_risk': disease_risk,
            'risk_factors': risk_factors,
            'optimal_range': f"{ranges['optimal_min']}-{ranges['optimal_max']}%"
        }
        
        return {
            'score': round(score, 2),
            'analysis': analysis
        }
    
    def _evaluate_solar_enhanced(self, solar: float) -> Dict[str, Any]:
        """Enhanced solar radiation evaluation"""
        ranges = self.optimal_ranges['solar']
        
        score = self._calculate_linear_score(solar, ranges)
        
        # Photosynthesis efficiency
        if ranges['optimal_min'] <= solar <= ranges['optimal_max']:
            photosynthesis_efficiency = "high"
        elif ranges['min'] <= solar <= ranges['max']:
            photosynthesis_efficiency = "medium"
        else:
            photosynthesis_efficiency = "low"
        
        analysis = {
            'value': solar,
            'photosynthesis_efficiency': photosynthesis_efficiency,
            'optimal_range': f"{ranges['optimal_min']}-{ranges['optimal_max']} MJ/m²/day"
        }
        
        return {
            'score': round(score, 2),
            'analysis': analysis
        }
    
    def _evaluate_wind_enhanced(self, wind_speed: float) -> Dict[str, Any]:
        """Enhanced wind speed evaluation"""
        ranges = self.optimal_ranges['wind_speed']
        
        score = self._calculate_linear_score(wind_speed, ranges)
        
        # Wind impact assessment
        if wind_speed > 6:
            impact = "high_damage_risk"
        elif wind_speed > 4:
            impact = "moderate_stress"
        elif wind_speed < 0.5:
            impact = "poor_air_circulation"
        else:
            impact = "beneficial"
        
        analysis = {
            'value': wind_speed,
            'impact': impact,
            'optimal_range': f"{ranges['optimal_min']}-{ranges['optimal_max']} m/s"
        }
        
        return {
            'score': round(score, 2),
            'analysis': analysis
        }
    
    def _calculate_linear_score(self, value: float, ranges: Dict) -> float:
        """Calculate linear suitability score"""
        if ranges['optimal_min'] <= value <= ranges['optimal_max']:
            return 100.0
        elif ranges['min'] <= value <= ranges['max']:
            if value < ranges['optimal_min']:
                return 60 + 40 * (value - ranges['min']) / (ranges['optimal_min'] - ranges['min'])
            else:
                return 60 + 40 * (ranges['max'] - value) / (ranges['max'] - ranges['optimal_max'])
        else:
            # Outside suitable range
            center = (ranges['min'] + ranges['max']) / 2
            return max(0, 30 - abs(value - center) / center * 30)
    
    def _calculate_gaussian_score(self, value: float, ranges: Dict) -> float:
        """Calculate Gaussian-based score for smoother transitions"""
        optimal_center = (ranges['optimal_min'] + ranges['optimal_max']) / 2
        optimal_width = ranges['optimal_max'] - optimal_center
        
        # Gaussian function centered at optimal range
        score = 100 * np.exp(-0.5 * ((value - optimal_center) / optimal_width) ** 2)
        
        return max(0, min(100, score))
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for param, score in component_scores.items():
            weight = self.weights.get(param, 0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _apply_quality_adjustments(self, base_score: float, data_quality: Dict, component_analysis: Dict) -> float:
        """Apply quality-based adjustments (similar to NASA preprocessing R² validation)"""
        
        adjusted_score = base_score
        
        # Data quality adjustment
        quality_factor = data_quality['quality_score'] / 100.0
        adjusted_score *= quality_factor
        
        # Risk-based adjustments
        high_risk_count = sum(
            1 for analysis in component_analysis['analysis'].values()
            if analysis.get('risk_level') in ['high', 'very_high']
        )
        
        if high_risk_count >= 2:
            # Multiple high-risk factors - additional penalty
            adjusted_score *= 0.9
        
        return max(0, min(100, adjusted_score))
    
    def _classify_suitability_enhanced(self, score: float) -> Dict[str, Any]:
        """Enhanced classification with confidence levels"""
        
        # Determine class
        if score >= self.quality_thresholds['excellent']:
            suitability_class = "Sangat Sesuai"
            confidence = "very_high"
            description = "Excellent conditions for rice cultivation"
        elif score >= self.quality_thresholds['good']:
            suitability_class = "Sesuai"
            confidence = "high"
            description = "Good conditions with minor limitations"
        elif score >= self.quality_thresholds['fair']:
            suitability_class = "Cukup Sesuai"
            confidence = "medium"
            description = "Moderately suitable with some constraints"
        elif score >= self.quality_thresholds['marginal']:
            suitability_class = "Kurang Sesuai"
            confidence = "low"
            description = "Marginal suitability, requires careful management"
        else:
            suitability_class = "Tidak Sesuai"
            confidence = "very_low"
            description = "Poor conditions, not recommended for rice cultivation"
        
        return {
            'class': suitability_class,
            'confidence': confidence,
            'description': description,
            'score_range': self._get_score_range(suitability_class)
        }
    
    def _assess_cultivation_risks(self, climate_data: Dict, component_analysis: Dict) -> Dict[str, Any]:
        """Assess cultivation risks based on climate analysis"""
        
        risks = {
            'drought_risk': 'low',
            'flood_risk': 'low',
            'disease_risk': 'low',
            'heat_stress_risk': 'low',
            'wind_damage_risk': 'low',
            'overall_risk': 'low'
        }
        
        # Analyze component risks
        for param, analysis in component_analysis['analysis'].items():
            if param == 'precipitation':
                if analysis['status'] == 'drought_risk':
                    risks['drought_risk'] = 'high'
                elif analysis['status'] == 'flood_risk':
                    risks['flood_risk'] = 'high'
                    
            elif param == 'humidity':
                risks['disease_risk'] = analysis.get('disease_risk', 'low')
                
            elif param == 'temperature':
                if analysis['risk_level'] == 'high':
                    risks['heat_stress_risk'] = 'high'
                    
            elif param == 'wind_speed':
                if analysis.get('impact') == 'high_damage_risk':
                    risks['wind_damage_risk'] = 'high'
        
        # Calculate overall risk
        high_risk_count = sum(1 for risk in risks.values() if risk == 'high')
        if high_risk_count >= 2:
            risks['overall_risk'] = 'high'
        elif high_risk_count == 1:
            risks['overall_risk'] = 'medium'
        
        return risks
    
    def _generate_recommendations(self, component_analysis: Dict, classification: Dict, risks: Dict) -> List[str]:
        """Generate cultivation recommendations"""
        
        recommendations = []
        
        # Temperature recommendations
        temp_analysis = component_analysis['analysis'].get('temperature', {})
        if temp_analysis.get('status') == 'critical':
            recommendations.append("Consider climate-controlled cultivation or select heat/cold-tolerant varieties")
        
        # Precipitation recommendations
        precip_analysis = component_analysis['analysis'].get('precipitation', {})
        if precip_analysis.get('status') == 'drought_risk':
            recommendations.append("Install irrigation systems and implement water conservation practices")
        elif precip_analysis.get('status') == 'flood_risk':
            recommendations.append("Implement proper drainage systems and consider flood-resistant varieties")
        
        # Humidity recommendations
        humidity_analysis = component_analysis['analysis'].get('humidity', {})
        if humidity_analysis.get('disease_risk') in ['high', 'very_high']:
            recommendations.append("Implement integrated pest management and improve field ventilation")
        
        # Overall recommendations based on classification
        if classification['confidence'] in ['low', 'very_low']:
            recommendations.append("Consider alternative crops or significant infrastructure investment")
        
        return recommendations
    
    def _extract_climate_values(self, climate_data: Dict[str, float]) -> Dict[str, float]:
        """Extract and format climate values for response"""
        return {
            'temperature': climate_data.get('T2M', 0),
            'precipitation_annual': climate_data.get('PRECTOTCORR', 0) * 365,
            'precipitation_daily': climate_data.get('PRECTOTCORR', 0),
            'humidity': climate_data.get('RH2M', 0),
            'solar_radiation': climate_data.get('ALLSKY_SFC_SW_DWN', 0),
            'wind_speed': climate_data.get('WS10M', 0)
        }
    
    def _get_score_range(self, suitability_class: str) -> str:
        """Get score range for suitability class"""
        ranges = {
            "Sangat Sesuai": "85-100",
            "Sesuai": "70-84",
            "Cukup Sesuai": "55-69",
            "Kurang Sesuai": "40-54",
            "Tidak Sesuai": "0-39"
        }
        return ranges.get(suitability_class, "Unknown")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'score': 0.0,
            'classification': 'Error',
            'confidence_level': 'none',
            'error': error_message,
            'component_scores': {},
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'method': 'enhanced_rice_suitability_v2',
                'status': 'error'
            }
        }
    
    def get_evaluation_parameters(self) -> Dict[str, Any]:
        """Get detailed evaluation parameters and thresholds"""
        return {
            'optimal_ranges': self.optimal_ranges,
            'weights': self.weights,
            'quality_thresholds': self.quality_thresholds,
            'evaluation_methods': ['linear_scoring', 'gaussian_smoothing', 'quality_adjustment'],
            'risk_factors': ['drought', 'flood', 'disease', 'heat_stress', 'wind_damage']
        }