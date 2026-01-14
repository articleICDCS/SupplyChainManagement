"""
Integration and Output Module
Consolidates insights from all modules into coherent decision-making outputs
as described in Section B.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .markov_chain_module import CompositeState, MarkovChainModule
from .probabilistic_forecasting_module import ProbabilisticForecastingModule, ForecastResult
from .dynamic_risk_assessment_module import DynamicRiskAssessmentModule, RiskAssessment
from .adaptive_thresholding_module import AdaptiveThresholdingModule, ClassificationResult, SegmentClassification

logger = logging.getLogger(__name__)


class RouteDecision(Enum):
    """Final route decision"""
    PROCEED = "proceed"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    REROUTE = "reroute"
    DELAY = "delay"
    ABORT = "abort"


@dataclass
class RouteSegment:
    """Road segment information"""
    segment_id: str
    length_km: float
    current_state: CompositeState
    risk_factors: Dict[str, float]


@dataclass
class OptimalRoute:
    """Optimal route recommendation"""
    route_id: str
    segments: List[str]
    total_distance_km: float
    total_risk_score: float
    temperature_stability_score: float
    utility_score: float
    decision: RouteDecision
    confidence: float
    reasoning: str
    contingency_plans: List[str]
    estimated_duration_hours: float


class IntegrationOutputModule:
    """
    Integration and Output Module for fusing information from all modules.
    
    Implements equation from paper:
    U(route) = α * (1 - R_route) + β * S_temp
    
    where R_route is aggregated risk, S_temp is temperature stability,
    and α, β are weighting factors balancing safety and cold-chain integrity.
    """
    
    def __init__(self,
                 markov_module: MarkovChainModule,
                 forecasting_module: ProbabilisticForecastingModule,
                 risk_assessment_module: DynamicRiskAssessmentModule,
                 thresholding_module: AdaptiveThresholdingModule,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Initialize Integration and Output Module
        
        Args:
            markov_module: Markov Chain Module
            forecasting_module: Probabilistic Forecasting Module
            risk_assessment_module: Dynamic Risk Assessment Module
            thresholding_module: Adaptive Thresholding Module
            alpha: Weight for safety (risk minimization)
            beta: Weight for cold-chain integrity (temperature stability)
        """
        self.markov_module = markov_module
        self.forecasting_module = forecasting_module
        self.risk_assessment_module = risk_assessment_module
        self.thresholding_module = thresholding_module
        
        # Utility function weights
        self.alpha = alpha  # Safety weight
        self.beta = beta    # Temperature stability weight
        
        # Ensure weights sum to 1
        total = alpha + beta
        self.alpha = alpha / total
        self.beta = beta / total
        
        logger.info(f"Integration Module initialized (α={self.alpha:.2f}, β={self.beta:.2f})")
    
    def evaluate_route(self,
                      route_segments: List[RouteSegment],
                      time_horizon: int = 4,
                      context: Optional[Dict] = None) -> OptimalRoute:
        """
        Evaluate a complete route by fusing all module outputs
        
        This is the main decision-making function that:
        1. Assesses risk for each segment
        2. Forecasts future states
        3. Classifies segments
        4. Computes utility score
        5. Makes final recommendation
        
        Args:
            route_segments: List of road segments in the route
            time_horizon: Forecast horizon (hours)
            context: Contextual information
            
        Returns:
            OptimalRoute with recommendation and reasoning
        """
        route_id = f"ROUTE_{'_'.join([s.segment_id for s in route_segments[:3]])}"
        
        # Step 1: Assess risk for each segment
        risk_assessments = []
        for segment in route_segments:
            risk_assessment = self.risk_assessment_module.assess_risk(
                segment_id=segment.segment_id,
                risk_factors=self._convert_to_risk_factors(segment.risk_factors),
                current_state=segment.current_state
            )
            risk_assessments.append(risk_assessment)
        
        # Step 2: Forecast future states for each segment
        forecasts = []
        for segment in route_segments:
            forecast = self.forecasting_module.forecast(
                current_state=segment.current_state,
                time_horizon=time_horizon,
                num_simulations=1000,
                context_data=context
            )
            forecasts.append(forecast)
        
        # Step 3: Classify each segment
        classifications = []
        for risk_assessment in risk_assessments:
            classification = self.thresholding_module.classify_segment(
                risk_assessment=risk_assessment,
                context=context
            )
            classifications.append(classification)
        
        # Step 4: Compute aggregated metrics
        total_risk = self._aggregate_route_risk(risk_assessments)
        temperature_stability = self._compute_temperature_stability(forecasts)
        
        # Step 5: Compute utility score: U(route) = α * (1 - R_route) + β * S_temp
        utility_score = self.alpha * (1.0 - total_risk) + self.beta * temperature_stability
        
        # Step 6: Make decision
        decision, confidence, reasoning = self._make_route_decision(
            classifications=classifications,
            utility_score=utility_score,
            total_risk=total_risk,
            temperature_stability=temperature_stability
        )
        
        # Step 7: Generate contingency plans
        contingency_plans = self._generate_contingency_plans(
            risk_assessments=risk_assessments,
            classifications=classifications
        )
        
        # Step 8: Estimate duration
        total_distance = sum(s.length_km for s in route_segments)
        estimated_duration = self._estimate_duration(total_distance, total_risk)
        
        logger.info(f"Route evaluated: {route_id}, Decision={decision.value}, "
                   f"Utility={utility_score:.3f}, Risk={total_risk:.3f}")
        
        return OptimalRoute(
            route_id=route_id,
            segments=[s.segment_id for s in route_segments],
            total_distance_km=total_distance,
            total_risk_score=total_risk,
            temperature_stability_score=temperature_stability,
            utility_score=utility_score,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            contingency_plans=contingency_plans,
            estimated_duration_hours=estimated_duration
        )
    
    def select_optimal_route(self,
                           candidate_routes: List[List[RouteSegment]],
                           context: Optional[Dict] = None) -> OptimalRoute:
        """
        Select the best route from multiple candidates
        
        Evaluates all routes and selects one with highest utility
        
        Args:
            candidate_routes: List of route candidates
            context: Contextual information
            
        Returns:
            Optimal route recommendation
        """
        evaluated_routes = []
        
        for route in candidate_routes:
            evaluated = self.evaluate_route(route, context=context)
            evaluated_routes.append(evaluated)
        
        # Select route with highest utility score
        optimal = max(evaluated_routes, key=lambda r: r.utility_score)
        
        logger.info(f"Optimal route selected: {optimal.route_id} "
                   f"(Utility={optimal.utility_score:.3f})")
        
        return optimal
    
    def _convert_to_risk_factors(self, factors_dict: Dict[str, float]):
        """Convert dictionary to RiskFactors object"""
        from .dynamic_risk_assessment_module import RiskFactors
        return RiskFactors(
            traffic_density=factors_dict.get('traffic_density', 0.3),
            route_security=factors_dict.get('route_security', 0.3),
            temperature_stability=factors_dict.get('temperature_stability', 0.3),
            weather_severity=factors_dict.get('weather_severity', 0.2),
            road_quality=factors_dict.get('road_quality', 0.3)
        )
    
    def _aggregate_route_risk(self, risk_assessments: List[RiskAssessment]) -> float:
        """
        Aggregate risk across all segments in route
        
        Uses weighted average with emphasis on highest-risk segments
        
        Args:
            risk_assessments: List of risk assessments
            
        Returns:
            Aggregated risk score R_route
        """
        if not risk_assessments:
            return 0.0
        
        risk_scores = [r.risk_score for r in risk_assessments]
        
        # Use combination of average and maximum to account for bottlenecks
        avg_risk = np.mean(risk_scores)
        max_risk = np.max(risk_scores)
        
        # Weighted combination: 60% average, 40% maximum
        aggregated_risk = 0.6 * avg_risk + 0.4 * max_risk
        
        return np.clip(aggregated_risk, 0.0, 1.0)
    
    def _compute_temperature_stability(self, forecasts: List[ForecastResult]) -> float:
        """
        Compute overall temperature stability score S_temp
        
        Higher score = more stable temperature conditions
        
        Args:
            forecasts: List of probabilistic forecasts
            
        Returns:
            Temperature stability score (0-1)
        """
        from .markov_chain_module import TemperatureCondition
        
        stability_scores = []
        
        for forecast in forecasts:
            # Calculate weighted stability based on predicted states
            stable_prob = sum(
                prob for state, prob in forecast.state_probabilities.items()
                if state.temperature_condition == TemperatureCondition.STABLE
            )
            
            unstable_prob = sum(
                prob for state, prob in forecast.state_probabilities.items()
                if state.temperature_condition == TemperatureCondition.UNSTABLE
            )
            
            critical_prob = sum(
                prob for state, prob in forecast.state_probabilities.items()
                if state.temperature_condition == TemperatureCondition.CRITICAL
            )
            
            # Compute stability score: 1.0 for stable, 0.5 for unstable, 0.0 for critical
            score = 1.0 * stable_prob + 0.5 * unstable_prob + 0.0 * critical_prob
            stability_scores.append(score)
        
        # Return average stability across all segments
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _make_route_decision(self,
                            classifications: List[ClassificationResult],
                            utility_score: float,
                            total_risk: float,
                            temperature_stability: float) -> Tuple[RouteDecision, float, str]:
        """
        Make final route decision based on all available information
        
        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        # Count segment classifications
        unavailable_count = sum(
            1 for c in classifications 
            if c.classification == SegmentClassification.UNAVAILABLE
        )
        conditional_count = sum(
            1 for c in classifications 
            if c.classification == SegmentClassification.CONDITIONAL
        )
        
        total_segments = len(classifications)
        
        # Decision logic
        if unavailable_count > 0:
            # Any unavailable segment -> cannot proceed
            decision = RouteDecision.ABORT
            confidence = 0.95
            reasoning = f"{unavailable_count}/{total_segments} segments unavailable - Route blocked"
        
        elif utility_score < 0.3:
            # Very low utility -> abort
            decision = RouteDecision.ABORT
            confidence = 0.90
            reasoning = f"Very low utility score ({utility_score:.2f}) - Unacceptable risk/temperature profile"
        
        elif utility_score < 0.5:
            # Low utility -> reroute or delay
            if conditional_count > total_segments / 2:
                decision = RouteDecision.DELAY
                confidence = 0.75
                reasoning = f"Low utility ({utility_score:.2f}), multiple conditional segments - Recommend delay"
            else:
                decision = RouteDecision.REROUTE
                confidence = 0.80
                reasoning = f"Low utility ({utility_score:.2f}) - Recommend alternative route"
        
        elif temperature_stability < 0.4:
            # Poor temperature stability
            decision = RouteDecision.PROCEED_WITH_CAUTION
            confidence = 0.70
            reasoning = f"Acceptable risk but poor temperature stability ({temperature_stability:.2f}) - Require enhanced monitoring"
        
        elif conditional_count > 0:
            # Some conditional segments
            decision = RouteDecision.PROCEED_WITH_CAUTION
            confidence = 0.80
            reasoning = f"{conditional_count} conditional segments - Proceed with monitoring and contingencies"
        
        else:
            # All clear
            decision = RouteDecision.PROCEED
            confidence = min(0.95, utility_score)
            reasoning = f"High utility score ({utility_score:.2f}) - Route approved"
        
        return decision, confidence, reasoning
    
    def _generate_contingency_plans(self,
                                   risk_assessments: List[RiskAssessment],
                                   classifications: List[ClassificationResult]) -> List[str]:
        """Generate contingency plans based on identified risks"""
        contingencies = []
        
        # Collect all recommendations from risk assessments
        for assessment in risk_assessments:
            contingencies.extend(assessment.recommendations)
        
        # Add classification-specific contingencies
        conditional_segments = [
            c.segment_id for c in classifications 
            if c.classification == SegmentClassification.CONDITIONAL
        ]
        
        if conditional_segments:
            contingencies.append(
                f"🔄 Monitor segments {', '.join(conditional_segments[:3])} closely - prepared for rerouting"
            )
        
        # Remove duplicates
        contingencies = list(set(contingencies))
        
        return contingencies[:10]  # Limit to top 10
    
    def _estimate_duration(self, distance_km: float, risk_score: float) -> float:
        """
        Estimate travel duration accounting for risk
        
        Higher risk = slower travel
        
        Args:
            distance_km: Total distance
            risk_score: Aggregated risk score
            
        Returns:
            Estimated duration in hours
        """
        # Base speed: 60 km/h
        base_speed = 60.0
        
        # Adjust speed based on risk (lower speed for higher risk)
        risk_penalty = 1.0 - (0.5 * risk_score)  # Up to 50% speed reduction
        effective_speed = base_speed * risk_penalty
        
        # Calculate duration
        duration = distance_km / effective_speed
        
        return duration
    
    def update_weights(self, new_alpha: float, new_beta: float):
        """
        Update utility function weights
        
        Args:
            new_alpha: New weight for safety
            new_beta: New weight for temperature stability
        """
        total = new_alpha + new_beta
        self.alpha = new_alpha / total
        self.beta = new_beta / total
        
        logger.info(f"Utility weights updated: α={self.alpha:.2f}, β={self.beta:.2f}")
    
    def compute_scri(self, routes: List[Dict], probabilities: List[float], costs: List[float]) -> float:
        """
        Compute Supply Chain Risk Index (SCRI)
        SCRI = sum(p_i * c_i) where p_i = probability, c_i = cost/severity
        
        Args:
            routes: List of route evaluations
            probabilities: Probability of risk event for each route
            costs: Cost/severity of failure for each route
            
        Returns:
            SCRI value (0-1, lower is better)
        """
        if len(probabilities) != len(costs):
            raise ValueError("Probabilities and costs must have same length")
        
        scri = sum(p * c for p, c in zip(probabilities, costs))
        return scri
    
    def compute_prediction_accuracy(self, predictions: List[bool], actuals: List[bool]) -> float:
        """
        Compute Prediction Accuracy (PA)
        PA = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            predictions: Predicted disruptions (True = disruption predicted)
            actuals: Actual disruptions (True = disruption occurred)
            
        Returns:
            PA value (0-1, higher is better)
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
        tn = sum(1 for p, a in zip(predictions, actuals) if not p and not a)
        fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
        fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)
        
        total = tp + tn + fp + fn
        if total == 0:
            return 0.0
        
        pa = (tp + tn) / total
        return pa
    
    def compute_adaptation_efficiency(self, system_time: float, baseline_time: float) -> float:
        """
        Compute Adaptation Efficiency (AE)
        AE = 1 - (T_system / T_baseline)
        
        Args:
            system_time: Reaction time of DT system (seconds)
            baseline_time: Reaction time of baseline method (seconds)
            
        Returns:
            AE value (0-1, higher is better)
        """
        if baseline_time <= 0:
            raise ValueError("Baseline time must be positive")
        
        ae = 1.0 - (system_time / baseline_time)
        return max(0.0, min(1.0, ae))  # Clamp to [0, 1]
    
    def compute_ccr(self, total_deliveries: int, temperature_violations: int) -> float:
        """
        Compute Cold-Chain Reliability Rate (CCR)
        CCR = 1 - (violations / total_deliveries)
        
        Args:
            total_deliveries: Total number of deliveries
            temperature_violations: Number of deliveries with temperature violations
            
        Returns:
            CCR value (0-1, higher is better)
        """
        if total_deliveries <= 0:
            raise ValueError("Total deliveries must be positive")
        
        ccr = 1.0 - (temperature_violations / total_deliveries)
        return max(0.0, min(1.0, ccr))  # Clamp to [0, 1]
