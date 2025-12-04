import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from datetime import datetime

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction classification"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NO_TREND = "no_trend"
    UNCERTAIN = "uncertain"

class TrendSignificance(Enum):
    """Statistical significance levels"""
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01
    SIGNIFICANT = "significant"                # p < 0.05
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    NOT_SIGNIFICANT = "not_significant"        # p >= 0.1

@dataclass
class TrendAnalysis:
    """Mann-Kendall trend analysis results"""
    parameter: str
    trend_direction: TrendDirection
    trend_magnitude: float
    significance: TrendSignificance
    p_value: float
    kendall_tau: float
    sen_slope: float
    projection_2030: float
    confidence_interval: Tuple[float, float]
    data_quality_score: float

@dataclass
class StabilityMetrics:
    """Temporal stability metrics"""
    parameter: str
    coefficient_of_variation: float
    stability_index: float
    anomaly_frequency: float
    recovery_rate: float
    trend_consistency: float

@dataclass
class SeasonalityAnalysis:
    """Seasonality analysis results"""
    parameter: str
    seasonal_strength: float
    peak_season: str
    seasonal_amplitude: float
    seasonal_consistency: float

@dataclass
class AnomalyDetection:
    """Anomaly detection results"""
    parameter: str
    anomaly_threshold: float
    detected_anomalies: List[Dict[str, Any]]
    anomaly_frequency: float
    severity_distribution: Dict[str, int]

class TemporalAnalysisService:
    """
    Comprehensive temporal analysis service for climate data
    Provides trend detection, stability metrics, seasonality analysis, and anomaly detection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.anomaly_threshold_multiplier = 2.0  # Standard deviations for anomaly detection
        self.min_data_points = 365  # Minimum data points for reliable analysis
        self.confidence_level = 0.95  # Confidence level for projections
        
        self.logger.info("Temporal Analysis Service initialized")
    
    def analyze_temporal_trends(self, 
                               climate_data: List[Dict[str, Any]], 
                               parameters: List[str] = None) -> Dict[str, TrendAnalysis]:
        """
        Perform Mann-Kendall trend analysis on climate parameters
        
        Args:
            climate_data: List of climate records with temporal data
            parameters: List of parameters to analyze (default: all available)
            
        Returns:
            Dictionary mapping parameter names to TrendAnalysis results
        """
        try:
            if not climate_data or len(climate_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for trend analysis: {len(climate_data) if climate_data else 0} points")
                return {}
            
            # Convert to DataFrame
            df = self._prepare_dataframe(climate_data)
            
            if df is None or len(df) == 0:
                return {}
            
            # Determine parameters to analyze
            if parameters is None:
                parameters = ['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M']
            
            available_params = [p for p in parameters if p in df.columns]
            
            if not available_params:
                self.logger.warning("No valid parameters found for trend analysis")
                return {}
            
            trend_results = {}
            
            for param in available_params:
                try:
                    trend_analysis = self._perform_mann_kendall_analysis(df, param)
                    trend_results[param] = trend_analysis
                    
                    self.logger.info(f"Trend analysis for {param}: "
                                   f"{trend_analysis.trend_direction.value} "
                                   f"(p={trend_analysis.p_value:.4f})")
                    
                except Exception as e:
                    self.logger.error(f"Error in trend analysis for {param}: {str(e)}")
                    continue
            
            return trend_results
            
        except Exception as e:
            self.logger.error(f"Error in temporal trends analysis: {str(e)}")
            return {}
    
    def calculate_stability_metrics(self, 
                                   climate_data: List[Dict[str, Any]], 
                                   parameters: List[str] = None) -> Dict[str, StabilityMetrics]:
        """
        Calculate temporal stability metrics for climate parameters
        
        Args:
            climate_data: List of climate records
            parameters: Parameters to analyze
            
        Returns:
            Dictionary mapping parameter names to StabilityMetrics
        """
        try:
            df = self._prepare_dataframe(climate_data)
            
            if df is None or len(df) == 0:
                return {}
            
            if parameters is None:
                parameters = ['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M']
            
            available_params = [p for p in parameters if p in df.columns]
            stability_results = {}
            
            for param in available_params:
                try:
                    stability_metrics = self._calculate_parameter_stability(df, param)
                    stability_results[param] = stability_metrics
                    
                    self.logger.info(f"Stability analysis for {param}: "
                                   f"CV={stability_metrics.coefficient_of_variation:.2f}, "
                                   f"Stability={stability_metrics.stability_index:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating stability for {param}: {str(e)}")
                    continue
            
            return stability_results
            
        except Exception as e:
            self.logger.error(f"Error in stability metrics calculation: {str(e)}")
            return {}
    
    def analyze_seasonality(self, 
                           climate_data: List[Dict[str, Any]], 
                           parameters: List[str] = None) -> Dict[str, SeasonalityAnalysis]:
        """
        Analyze seasonal patterns in climate data
        
        Args:
            climate_data: List of climate records
            parameters: Parameters to analyze
            
        Returns:
            Dictionary mapping parameter names to SeasonalityAnalysis
        """
        try:
            df = self._prepare_dataframe(climate_data)
            
            if df is None or len(df) == 0:
                return {}
            
            if parameters is None:
                parameters = ['T2M', 'PRECTOTCORR', 'RH2M']
            
            available_params = [p for p in parameters if p in df.columns]
            seasonality_results = {}
            
            for param in available_params:
                try:
                    seasonality_analysis = self._analyze_parameter_seasonality(df, param)
                    seasonality_results[param] = seasonality_analysis
                    
                    self.logger.info(f"Seasonality analysis for {param}: "
                                   f"Peak season={seasonality_analysis.peak_season}, "
                                   f"Strength={seasonality_analysis.seasonal_strength:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in seasonality analysis for {param}: {str(e)}")
                    continue
            
            return seasonality_results
            
        except Exception as e:
            self.logger.error(f"Error in seasonality analysis: {str(e)}")
            return {}
    
    def detect_anomalies(self, 
                        climate_data: List[Dict[str, Any]], 
                        parameters: List[str] = None) -> Dict[str, AnomalyDetection]:
        """
        Detect anomalous events in climate data
        
        Args:
            climate_data: List of climate records
            parameters: Parameters to analyze
            
        Returns:
            Dictionary mapping parameter names to AnomalyDetection results
        """
        try:
            df = self._prepare_dataframe(climate_data)
            
            if df is None or len(df) == 0:
                return {}
            
            if parameters is None:
                parameters = ['T2M', 'PRECTOTCORR', 'RH2M']
            
            available_params = [p for p in parameters if p in df.columns]
            anomaly_results = {}
            
            for param in available_params:
                try:
                    anomaly_detection = self._detect_parameter_anomalies(df, param)
                    anomaly_results[param] = anomaly_detection
                    
                    self.logger.info(f"Anomaly detection for {param}: "
                                   f"{len(anomaly_detection.detected_anomalies)} anomalies detected")
                    
                except Exception as e:
                    self.logger.error(f"Error in anomaly detection for {param}: {str(e)}")
                    continue
            
            return anomaly_results
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {}
    
    def analyze_temporal_stability(self, climate_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Comprehensive temporal stability analysis (for food security analyzer integration)
        
        Args:
            climate_data: List of climate records
            
        Returns:
            Dictionary with stability metrics for food security analysis
        """
        try:
            stability_metrics = self.calculate_stability_metrics(climate_data)
            
            if not stability_metrics:
                return self._default_stability_metrics()
            
            # Calculate overall stability scores
            overall_stability = 0
            climate_variability = 0
            trend_consistency = 0
            anomaly_resilience = 0
            
            param_count = len(stability_metrics)
            
            for param, metrics in stability_metrics.items():
                # Convert CV to stability (lower CV = higher stability)
                param_stability = max(0, 100 - (metrics.coefficient_of_variation * 2))
                overall_stability += param_stability
                
                # Climate variability (CV)
                climate_variability += metrics.coefficient_of_variation
                
                # Trend consistency
                trend_consistency += metrics.trend_consistency
                
                # Anomaly resilience (inverse of anomaly frequency)
                resilience = max(0, 100 - (metrics.anomaly_frequency * 100))
                anomaly_resilience += resilience
            
            if param_count > 0:
                overall_stability /= param_count
                climate_variability /= param_count
                trend_consistency /= param_count
                anomaly_resilience /= param_count
            
            return {
                'overall_stability': overall_stability,
                'climate_variability': climate_variability,
                'trend_consistency': trend_consistency,
                'anomaly_resilience': anomaly_resilience
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal stability analysis: {str(e)}")
            return self._default_stability_metrics()
    
    def _prepare_dataframe(self, climate_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame from climate data with proper time index"""
        try:
            if not climate_data:
                return None
            
            df = pd.DataFrame(climate_data)
            
            # Ensure date column exists and convert to datetime
            if 'date' not in df.columns:
                # Create date range if no date column
                start_date = '2005-01-01'
                df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
                self.logger.warning("No date column found, created synthetic date range")
            else:
                df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Remove non-numeric columns for analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def _perform_mann_kendall_analysis(self, df: pd.DataFrame, parameter: str) -> TrendAnalysis:
        """Perform Mann-Kendall trend test on a parameter"""
        try:
            data = df[parameter].dropna()
            
            if len(data) < 10:
                raise ValueError(f"Insufficient data points for {parameter}: {len(data)}")
            
            # Mann-Kendall test using scipy.stats
            n = len(data)
            
            # Calculate Kendall's tau
            tau, p_value = stats.kendalltau(range(n), data.values)
            
            # Sen's slope estimator
            sen_slope = self._calculate_sen_slope(data.values)
            
            # Determine trend direction
            if p_value < 0.05:
                if tau > 0:
                    trend_direction = TrendDirection.INCREASING
                else:
                    trend_direction = TrendDirection.DECREASING
            else:
                trend_direction = TrendDirection.NO_TREND
            
            # Determine significance
            if p_value < 0.01:
                significance = TrendSignificance.HIGHLY_SIGNIFICANT
            elif p_value < 0.05:
                significance = TrendSignificance.SIGNIFICANT
            elif p_value < 0.1:
                significance = TrendSignificance.MARGINALLY_SIGNIFICANT
            else:
                significance = TrendSignificance.NOT_SIGNIFICANT
            
            # Calculate trend magnitude (annual change)
            trend_magnitude = abs(sen_slope * 365)  # Daily to annual
            
            # Project to 2030 (assuming current year is ~2024)
            years_to_project = 6
            current_mean = data.mean()
            projection_2030 = current_mean + (sen_slope * 365 * years_to_project)
            
            # Calculate confidence interval (simplified)
            std_error = data.std() / np.sqrt(len(data))
            margin_error = 1.96 * std_error  # 95% confidence
            confidence_interval = (
                projection_2030 - margin_error,
                projection_2030 + margin_error
            )
            
            # Data quality score based on completeness and consistency
            data_quality_score = min(100, (len(data) / len(df)) * 100)
            
            return TrendAnalysis(
                parameter=parameter,
                trend_direction=trend_direction,
                trend_magnitude=trend_magnitude,
                significance=significance,
                p_value=p_value,
                kendall_tau=tau,
                sen_slope=sen_slope,
                projection_2030=projection_2030,
                confidence_interval=confidence_interval,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error in Mann-Kendall analysis for {parameter}: {str(e)}")
            raise
    
    def _calculate_sen_slope(self, data: np.ndarray) -> float:
        """Calculate Sen's slope estimator"""
        try:
            n = len(data)
            slopes = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if j != i:
                        slope = (data[j] - data[i]) / (j - i)
                        slopes.append(slope)
            
            if slopes:
                return np.median(slopes)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating Sen's slope: {str(e)}")
            return 0.0
    
    def _calculate_parameter_stability(self, df: pd.DataFrame, parameter: str) -> StabilityMetrics:
        """Calculate stability metrics for a parameter"""
        try:
            data = df[parameter].dropna()
            
            # Coefficient of variation
            cv = (data.std() / data.mean()) * 100 if data.mean() != 0 else 100
            
            # Stability index (inverse of CV, normalized to 0-100)
            stability_index = max(0, 100 - cv)
            
            # Anomaly frequency (values beyond 2 standard deviations)
            mean_val = data.mean()
            std_val = data.std()
            anomalies = data[abs(data - mean_val) > 2 * std_val]
            anomaly_frequency = len(anomalies) / len(data)
            
            # Recovery rate (how quickly values return to normal after anomalies)
            recovery_rate = self._calculate_recovery_rate(data, mean_val, std_val)
            
            # Trend consistency (how consistent the trend is over time)
            trend_consistency = self._calculate_trend_consistency(data)
            
            return StabilityMetrics(
                parameter=parameter,
                coefficient_of_variation=cv,
                stability_index=stability_index,
                anomaly_frequency=anomaly_frequency,
                recovery_rate=recovery_rate,
                trend_consistency=trend_consistency
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating stability metrics for {parameter}: {str(e)}")
            raise
    
    def _calculate_recovery_rate(self, data: pd.Series, mean_val: float, std_val: float) -> float:
        """Calculate recovery rate from anomalies"""
        try:
            # Simplified recovery rate calculation
            # Count how often values return to within 1 std after being beyond 2 std
            threshold = 2 * std_val
            normal_threshold = 1 * std_val
            
            recovery_events = 0
            anomaly_events = 0
            in_anomaly = False
            
            for value in data:
                deviation = abs(value - mean_val)
                
                if deviation > threshold and not in_anomaly:
                    in_anomaly = True
                    anomaly_events += 1
                elif deviation <= normal_threshold and in_anomaly:
                    in_anomaly = False
                    recovery_events += 1
            
            if anomaly_events > 0:
                return (recovery_events / anomaly_events) * 100
            else:
                return 100.0  # No anomalies = perfect recovery
                
        except Exception as e:
            self.logger.error(f"Error calculating recovery rate: {str(e)}")
            return 75.0
    
    def _calculate_trend_consistency(self, data: pd.Series) -> float:
        """Calculate how consistent the trend is over different time windows"""
        try:
            # Split data into quarters and check trend consistency
            quarter_size = len(data) // 4
            if quarter_size < 10:
                return 70.0  # Not enough data for reliable consistency measure
            
            trend_directions = []
            
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = start_idx + quarter_size
                quarter_data = data.iloc[start_idx:end_idx]
                
                # Simple trend direction using first and last values
                if len(quarter_data) >= 2:
                    trend = quarter_data.iloc[-1] - quarter_data.iloc[0]
                    trend_directions.append(1 if trend > 0 else -1 if trend < 0 else 0)
            
            if not trend_directions:
                return 70.0
            
            # Calculate consistency (how often trends agree)
            most_common_trend = max(set(trend_directions), key=trend_directions.count)
            consistency = (trend_directions.count(most_common_trend) / len(trend_directions)) * 100
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {str(e)}")
            return 70.0
    
    def _analyze_parameter_seasonality(self, df: pd.DataFrame, parameter: str) -> SeasonalityAnalysis:
        """Analyze seasonal patterns in a parameter"""
        try:
            data = df[parameter].dropna()
            
            # Extract month from index
            if hasattr(data.index, 'month'):
                monthly_data = data.groupby(data.index.month).mean()
            else:
                # Fallback: assume daily data and use modulo
                monthly_data = data.groupby(data.index % 12).mean()
            
            # Calculate seasonal strength
            overall_mean = data.mean()
            seasonal_variance = monthly_data.var()
            total_variance = data.var()
            seasonal_strength = (seasonal_variance / total_variance) * 100 if total_variance > 0 else 0
            
            # Find peak season
            peak_month = monthly_data.idxmax()
            season_names = {
                12: "Dry Season", 1: "Dry Season", 2: "Dry Season",
                3: "Transition", 4: "Transition", 
                5: "Wet Season", 6: "Wet Season", 7: "Wet Season", 
                8: "Wet Season", 9: "Wet Season",
                10: "Transition", 11: "Transition"
            }
            peak_season = season_names.get(peak_month, "Unknown")
            
            # Seasonal amplitude
            seasonal_amplitude = monthly_data.max() - monthly_data.min()
            
            # Seasonal consistency (how consistent the pattern is year over year)
            seasonal_consistency = min(100, seasonal_strength)  # Simplified
            
            return SeasonalityAnalysis(
                parameter=parameter,
                seasonal_strength=seasonal_strength,
                peak_season=peak_season,
                seasonal_amplitude=seasonal_amplitude,
                seasonal_consistency=seasonal_consistency
            )
            
        except Exception as e:
            self.logger.error(f"Error in seasonality analysis for {parameter}: {str(e)}")
            raise
    
    def _detect_parameter_anomalies(self, df: pd.DataFrame, parameter: str) -> AnomalyDetection:
        """Detect anomalies in a parameter"""
        try:
            data = df[parameter].dropna()
            
            # Calculate thresholds
            mean_val = data.mean()
            std_val = data.std()
            threshold = self.anomaly_threshold_multiplier * std_val
            
            # Detect anomalies
            anomalies = []
            for date, value in data.items():
                deviation = abs(value - mean_val)
                if deviation > threshold:
                    severity = "extreme" if deviation > 3 * std_val else "moderate"
                    anomalies.append({
                        'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                        'value': value,
                        'deviation': deviation,
                        'severity': severity,
                        'z_score': (value - mean_val) / std_val if std_val > 0 else 0
                    })
            
            # Calculate anomaly frequency
            anomaly_frequency = len(anomalies) / len(data)
            
            # Severity distribution
            severity_distribution = {
                'moderate': len([a for a in anomalies if a['severity'] == 'moderate']),
                'extreme': len([a for a in anomalies if a['severity'] == 'extreme'])
            }
            
            return AnomalyDetection(
                parameter=parameter,
                anomaly_threshold=threshold,
                detected_anomalies=anomalies,
                anomaly_frequency=anomaly_frequency,
                severity_distribution=severity_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection for {parameter}: {str(e)}")
            raise
    
    def _default_stability_metrics(self) -> Dict[str, float]:
        """Return default stability metrics when analysis fails"""
        return {
            'overall_stability': 75.0,
            'climate_variability': 25.0,
            'trend_consistency': 70.0,
            'anomaly_resilience': 75.0
        }