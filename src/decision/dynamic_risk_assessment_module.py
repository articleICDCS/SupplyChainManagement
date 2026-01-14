"""
Dynamic Risk Assessment Module
Implements weighted scoring function for evaluating risk of road segments
as described in Section B.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from .markov_chain_module import CompositeState, RoadCondition, TemperatureCondition

logger = logging.getLogger(__name__)


@dataclass
class RiskFactors:
    """Risk factors for a road segment"""
    traffic_density: float  # 0-1 scale
    route_security: float   # 0-1 scale (1 = very insecure)
    temperature_stability: float  # 0-1 scale (1 = very unstable)
    weather_severity: float  # 0-1 scale
    road_quality: float  # 0-1 scale (1 = very poor)


@dataclass
class RiskAssessment:
    """Result of risk assessment"""
    segment_id: str
    risk_score: float  # R_segment
    risk_category: str  # low, medium, high, critical
    contributing_factors: Dict[str, float]
    recommendations: List[str]


class DynamicRiskAssessmentModule:
    """
    Dynamic Risk Assessment Module with adaptive weighting.
    
    Implements equation from paper:
    R_segment = Σ w_i * f_i
    
    where f_i represents factors (traffic, security, temperature, etc.)
    and w_i are adaptive weights normalized to reflect current operational context.
    """
    
    def __init__(self, operational_context: str = "normal"):
        """
        Initialize Dynamic Risk Assessment Module
        
        Args:
            operational_context: Operating context - "normal", "wartime", "pandemic", "crisis"
        """
        self.operational_context = operational_context
        self.weights = self._initialize_weights(operational_context)
        
        logger.info(f"Risk Assessment initialized for context: {operational_context}")
    
    def _initialize_weights(self, context: str) -> Dict[str, float]:
        """
        Initialize adaptive weights based on operational context
        
        As stated in paper: "during wartime contexts, road security disruptions 
        may dominate the risk profile, while in cold-chain distribution, 
        transitions toward unstable or critical temperature states significantly 
        increase the risk score"
        
        Args:
            context: Operational context
            
        Returns:
            Dictionary of normalized weights
        """
        if context == "wartime":
            weights = {
                'traffic_density': 0.10,
                'route_security': 0.45,  # Dominant in wartime
                'temperature_stability': 0.25,
                'weather_severity': 0.10,
                'road_quality': 0.10
            }
        elif context == "pandemic":
            weights = {
                'traffic_density': 0.20,
                'route_security': 0.15,
                'temperature_stability': 0.40,  # Critical for vaccines
                'weather_severity': 0.15,
                'road_quality': 0.10
            }
        elif context == "crisis":
            weights = {
                'traffic_density': 0.15,
                'route_security': 0.30,
                'temperature_stability': 0.35,
                'weather_severity': 0.15,
                'road_quality': 0.05
            }
        else:  # normal
            weights = {
                'traffic_density': 0.25,
                'route_security': 0.20,
                'temperature_stability': 0.25,
                'weather_severity': 0.15,
                'road_quality': 0.15
            }
        
        # Normalize to ensure sum = 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def assess_risk(self,
                   segment_id: str,
                   risk_factors: RiskFactors,
                   current_state: Optional[CompositeState] = None) -> RiskAssessment:
        """
        Assess risk of a road segment using weighted scoring function
        
        Implements: R_segment = Σ w_i * f_i
        
        Args:
            segment_id: Road segment identifier
            risk_factors: RiskFactors object with all factor values
            current_state: Current composite state (if available)
            
        Returns:
            RiskAssessment with score and recommendations
        """
        # Extract factor values
        factors = {
            'traffic_density': risk_factors.traffic_density,
            'route_security': risk_factors.route_security,
            'temperature_stability': risk_factors.temperature_stability,
            'weather_severity': risk_factors.weather_severity,
            'road_quality': risk_factors.road_quality
        }
        
        # Adjust factors based on current state if provided
        if current_state:
            factors = self._adjust_factors_by_state(factors, current_state)
        
        # Calculate weighted risk score: R_segment = Σ w_i * f_i
        risk_score = sum(self.weights[key] * factors[key] for key in self.weights.keys())
        
        # Clamp to [0, 1]
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Categorize risk
        risk_category = self._categorize_risk(risk_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_score, factors, current_state)
        
        # Identify contributing factors (those above threshold)
        contributing_factors = {
            key: factors[key] 
            for key in factors.keys() 
            if factors[key] > 0.5
        }
        
        logger.debug(f"Risk assessment for {segment_id}: score={risk_score:.3f}, category={risk_category}")
        
        return RiskAssessment(
            segment_id=segment_id,
            risk_score=risk_score,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def _adjust_factors_by_state(self, 
                                 factors: Dict[str, float],
                                 state: CompositeState) -> Dict[str, float]:
        """
        Adjust factor values based on current composite state
        
        As stated in paper: "transitions toward unstable or critical temperature 
        states significantly increase the risk score, even if the road remains 
        physically available"
        
        Args:
            factors: Original factor values
            state: Current composite state
            
        Returns:
            Adjusted factor values
        """
        adjusted = factors.copy()
        
        # Temperature state adjustments
        if state.temperature_condition == TemperatureCondition.UNSTABLE:
            adjusted['temperature_stability'] = max(adjusted['temperature_stability'], 0.6)
        elif state.temperature_condition == TemperatureCondition.CRITICAL:
            adjusted['temperature_stability'] = max(adjusted['temperature_stability'], 0.9)
        
        # Road state adjustments
        if state.road_condition == RoadCondition.DANGEROUS:
            adjusted['route_security'] = max(adjusted['route_security'], 0.7)
        elif state.road_condition == RoadCondition.UNAVAILABLE:
            adjusted['route_security'] = 1.0
            adjusted['road_quality'] = 1.0
        
        return adjusted
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into discrete levels"""
        if risk_score < 0.25:
            return "low"
        elif risk_score < 0.50:
            return "medium"
        elif risk_score < 0.75:
            return "high"
        else:
            return "critical"
    
    def _generate_recommendations(self,
                                 risk_score: float,
                                 factors: Dict[str, float],
                                 state: Optional[CompositeState]) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        # High overall risk
        if risk_score > 0.75:
            recommendations.append("⚠️ Critical risk level - Consider alternative route")
        
        # Temperature issues
        if factors['temperature_stability'] > 0.6:
            recommendations.append("🌡️ Temperature instability detected - Check refrigeration system")
        
        if state and state.temperature_condition == TemperatureCondition.CRITICAL:
            recommendations.append("🔴 CRITICAL: Temperature exceeds safe limits - Immediate action required")
        
        # Security issues
        if factors['route_security'] > 0.6:
            recommendations.append("🛡️ High security risk - Require escort or delay transport")
        
        if state and state.road_condition == RoadCondition.DANGEROUS:
            recommendations.append("⚠️ Dangerous road conditions - Proceed with extreme caution")
        
        # Traffic issues
        if factors['traffic_density'] > 0.7:
            recommendations.append("🚦 Heavy traffic - Consider alternative timing or route")
        
        # Weather issues
        if factors['weather_severity'] > 0.7:
            recommendations.append("🌩️ Severe weather conditions - Monitor forecasts closely")
        
        # Road quality
        if factors['road_quality'] > 0.7:
            recommendations.append("🛣️ Poor road conditions - Use specialized vehicle")
        
        return recommendations
    
    def update_operational_context(self, new_context: str):
        """
        Update operational context and recalculate weights
        
        Allows dynamic adaptation as situation evolves
        
        Args:
            new_context: New operational context
        """
        self.operational_context = new_context
        self.weights = self._initialize_weights(new_context)
        logger.info(f"Operational context updated to: {new_context}")
    
    def get_adaptive_weights(self, 
                            current_priority: str,
                            custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get adaptive weights based on current priority
        
        Allows fine-tuning for specific situations
        
        Args:
            current_priority: Priority focus - "speed", "safety", "temperature", "cost"
            custom_weights: Optional custom weight overrides
            
        Returns:
            Adjusted weights dictionary
        """
        if custom_weights:
            # Normalize custom weights
            total = sum(custom_weights.values())
            return {k: v/total for k, v in custom_weights.items()}
        
        # Adjust based on priority
        adjusted_weights = self.weights.copy()
        
        if current_priority == "temperature":
            adjusted_weights['temperature_stability'] *= 1.5
        elif current_priority == "safety":
            adjusted_weights['route_security'] *= 1.5
        elif current_priority == "speed":
            adjusted_weights['traffic_density'] *= 1.5
            adjusted_weights['road_quality'] *= 1.3
        
        # Renormalize
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
