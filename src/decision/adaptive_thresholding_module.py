"""
Adaptive Thresholding Module
Implements dynamic classification of road segments using adaptive decision tree
as described in Section B.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .dynamic_risk_assessment_module import RiskAssessment

logger = logging.getLogger(__name__)


class SegmentClassification(Enum):
    """Classification of road segments"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    CONDITIONAL = "conditional"  # Available with restrictions


@dataclass
class ClassificationResult:
    """Result of segment classification"""
    segment_id: str
    classification: SegmentClassification
    confidence: float
    threshold_used: float
    reasoning: str


class AdaptiveThresholdingModule:
    """
    Adaptive Thresholding Module for dynamic classification.
    
    Implements equation from paper:
    Class(segment) = {
        Available,    if R_segment < θ_t
        Unavailable,  if R_segment ≥ θ_t
    }
    
    where θ_t evolves over time as new contextual and sensor data become available.
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.5,
                 adaptive: bool = True,
                 boosting_enabled: bool = True):
        """
        Initialize Adaptive Thresholding Module
        
        Args:
            initial_threshold: Initial decision threshold θ_0
            adaptive: Enable threshold adaptation over time
            boosting_enabled: Enable boosting techniques for improved accuracy
        """
        self.threshold = initial_threshold  # θ_t
        self.adaptive = adaptive
        self.boosting_enabled = boosting_enabled
        
        # History for learning
        self.classification_history: List[Tuple[float, bool]] = []  # (risk_score, was_actually_unavailable)
        self.threshold_history: List[float] = [initial_threshold]
        
        # Boosting parameters
        self.sample_weights = []  # For misclassified samples
        
        logger.info(f"Adaptive Thresholding initialized with θ_0={initial_threshold}")
    
    def classify_segment(self,
                        risk_assessment: RiskAssessment,
                        context: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify road segment based on adaptive threshold
        
        Implements the decision rule from paper with real-time adaptability
        
        Args:
            risk_assessment: RiskAssessment from Dynamic Risk Assessment Module
            context: Optional contextual information for threshold adjustment
            
        Returns:
            ClassificationResult with classification and reasoning
        """
        risk_score = risk_assessment.risk_score
        
        # Apply contextual threshold adjustment if provided
        current_threshold = self._get_adjusted_threshold(context)
        
        # Classification decision
        if risk_score < current_threshold:
            classification = SegmentClassification.AVAILABLE
            confidence = 1.0 - (risk_score / current_threshold)
        else:
            # Check if conditionally available (with restrictions)
            if risk_score < current_threshold + 0.15:
                classification = SegmentClassification.CONDITIONAL
                confidence = 0.7
            else:
                classification = SegmentClassification.UNAVAILABLE
                confidence = (risk_score - current_threshold) / (1.0 - current_threshold)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(risk_score, current_threshold, 
                                            risk_assessment.risk_category)
        
        logger.debug(f"Classified {risk_assessment.segment_id}: {classification.value} "
                    f"(R={risk_score:.3f}, θ={current_threshold:.3f})")
        
        return ClassificationResult(
            segment_id=risk_assessment.segment_id,
            classification=classification,
            confidence=confidence,
            threshold_used=current_threshold,
            reasoning=reasoning
        )
    
    def _get_adjusted_threshold(self, context: Optional[Dict]) -> float:
        """
        Get adjusted threshold based on context
        
        Allows temporary threshold adjustments without permanent changes
        
        Args:
            context: Contextual information
            
        Returns:
            Adjusted threshold value
        """
        adjusted_threshold = self.threshold
        
        if context:
            # Adjust threshold based on urgency
            if context.get('urgency') == 'high':
                # More lenient threshold for urgent deliveries
                adjusted_threshold += 0.1
            
            # Adjust based on cargo criticality
            if context.get('cargo_critical'):
                # Stricter threshold for critical cargo
                adjusted_threshold -= 0.1
            
            # Adjust based on time of day
            if context.get('time_of_day') == 'night':
                # Stricter threshold at night (higher risk)
                adjusted_threshold -= 0.05
        
        # Clamp to valid range
        adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.9)
        
        return adjusted_threshold
    
    def _generate_reasoning(self, 
                           risk_score: float,
                           threshold: float,
                           risk_category: str) -> str:
        """Generate human-readable reasoning for classification"""
        if risk_score < threshold:
            margin = threshold - risk_score
            if margin > 0.3:
                return f"Low risk (R={risk_score:.2f}) - Well below threshold - Safe to proceed"
            else:
                return f"Acceptable risk (R={risk_score:.2f}) - Below threshold - Proceed with monitoring"
        else:
            excess = risk_score - threshold
            if excess < 0.15:
                return f"Marginal risk (R={risk_score:.2f}) - Slightly above threshold - Conditional approval possible"
            elif excess < 0.3:
                return f"High risk (R={risk_score:.2f}) - Above threshold - Recommend alternative"
            else:
                return f"Critical risk (R={risk_score:.2f}) - Well above threshold - Route unavailable"
    
    def update_threshold_from_feedback(self, 
                                      predicted_available: bool,
                                      actual_available: bool,
                                      risk_score: float):
        """
        Update threshold based on feedback (actual outcomes vs predictions)
        
        Implements adaptive learning: θ_t evolves as new data become available
        
        This is the core mechanism mentioned in paper: "The threshold θ_t evolves 
        over time as new contextual and sensor data become available"
        
        Args:
            predicted_available: What the model predicted
            actual_available: What actually happened
            risk_score: Risk score that was used for prediction
        """
        if not self.adaptive:
            return
        
        # Record in history
        self.classification_history.append((risk_score, not actual_available))
        
        # Check for misclassification
        misclassified = (predicted_available != actual_available)
        
        if misclassified:
            if predicted_available and not actual_available:
                # False positive: predicted available but was actually unavailable
                # Threshold was too high - decrease it
                adjustment = -0.02
                logger.debug(f"False positive detected - Lowering threshold by {abs(adjustment)}")
            else:
                # False negative: predicted unavailable but was actually available
                # Threshold was too low - increase it
                adjustment = 0.02
                logger.debug(f"False negative detected - Raising threshold by {adjustment}")
            
            # Apply adjustment
            self.threshold = np.clip(self.threshold + adjustment, 0.1, 0.9)
            self.threshold_history.append(self.threshold)
        
        # Apply boosting if enabled
        if self.boosting_enabled and misclassified:
            self._apply_boosting_update(risk_score, actual_available)
    
    def _apply_boosting_update(self, risk_score: float, actual_unavailable: bool):
        """
        Apply boosting technique to improve classification on difficult cases
        
        Implements adaptive decision tree with boosting as mentioned in paper
        
        Args:
            risk_score: Risk score of misclassified example
            actual_unavailable: True label
        """
        # Add sample weight for this difficult case
        self.sample_weights.append({
            'risk_score': risk_score,
            'label': actual_unavailable,
            'weight': 1.5  # Higher weight for misclassified samples
        })
        
        # Keep only recent samples (sliding window)
        if len(self.sample_weights) > 100:
            self.sample_weights = self.sample_weights[-100:]
        
        # Recompute threshold using weighted samples
        if len(self.sample_weights) > 20:
            self._recompute_threshold_with_boosting()
    
    def _recompute_threshold_with_boosting(self):
        """
        Recompute optimal threshold using boosted samples
        
        Finds threshold that minimizes weighted classification error
        """
        # Extract data
        scores = [s['risk_score'] for s in self.sample_weights]
        labels = [s['label'] for s in self.sample_weights]
        weights = [s['weight'] for s in self.sample_weights]
        
        # Try different threshold values
        candidate_thresholds = np.linspace(0.2, 0.8, 30)
        best_threshold = self.threshold
        min_error = float('inf')
        
        for theta in candidate_thresholds:
            # Calculate weighted error
            error = 0.0
            for score, label, weight in zip(scores, labels, weights):
                predicted = (score >= theta)  # Predict unavailable if score >= theta
                if predicted != label:
                    error += weight
            
            if error < min_error:
                min_error = error
                best_threshold = theta
        
        # Update threshold if significantly better
        if abs(best_threshold - self.threshold) > 0.05:
            logger.info(f"Boosting update: threshold {self.threshold:.3f} -> {best_threshold:.3f}")
            self.threshold = best_threshold
            self.threshold_history.append(best_threshold)
    
    def get_threshold_statistics(self) -> Dict:
        """Get statistics about threshold evolution"""
        if len(self.threshold_history) < 2:
            return {
                'current_threshold': self.threshold,
                'initial_threshold': self.threshold_history[0],
                'num_updates': 0,
                'average_threshold': self.threshold
            }
        
        return {
            'current_threshold': self.threshold,
            'initial_threshold': self.threshold_history[0],
            'num_updates': len(self.threshold_history) - 1,
            'average_threshold': np.mean(self.threshold_history),
            'threshold_std': np.std(self.threshold_history),
            'min_threshold': min(self.threshold_history),
            'max_threshold': max(self.threshold_history)
        }
    
    def reset_threshold(self, new_threshold: Optional[float] = None):
        """
        Reset threshold to initial value or new value
        
        Args:
            new_threshold: New threshold value (or use initial if None)
        """
        if new_threshold is not None:
            self.threshold = np.clip(new_threshold, 0.1, 0.9)
        else:
            self.threshold = self.threshold_history[0]
        
        logger.info(f"Threshold reset to {self.threshold:.3f}")
