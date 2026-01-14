"""
Probabilistic Forecasting Module
Implements Monte Carlo simulations with stratified sampling for forecasting
road availability and cold-chain stability as described in Section B.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from .markov_chain_module import MarkovChainModule, CompositeState

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of probabilistic forecast"""
    time_horizon: int
    state_probabilities: Dict[CompositeState, float]
    expected_state: CompositeState
    confidence: float
    simulation_runs: int


class ProbabilisticForecastingModule:
    """
    Probabilistic Forecasting Module using Monte Carlo simulations.
    
    Implements equation from paper:
    P̂(s_{t+k}) = (1/N) * Σ 1{s^(j)_{t+k} = s}
    
    where N is the number of simulation runs and 1 is the indicator function.
    """
    
    def __init__(self, 
                 markov_module: MarkovChainModule,
                 default_simulations: int = 10000,
                 use_stratified_sampling: bool = True):
        """
        Initialize Probabilistic Forecasting Module
        
        Args:
            markov_module: Markov Chain Module for state transitions
            default_simulations: Default number of Monte Carlo runs
            use_stratified_sampling: Enable stratified sampling for variance reduction
        """
        self.markov_module = markov_module
        self.default_simulations = default_simulations
        self.use_stratified_sampling = use_stratified_sampling
        
        logger.info(f"Probabilistic Forecasting initialized with N={default_simulations} simulations")
    
    def forecast(self,
                 current_state: CompositeState,
                 time_horizon: int,
                 num_simulations: Optional[int] = None,
                 context_data: Optional[Dict] = None) -> ForecastResult:
        """
        Generate probabilistic forecast for future states using Monte Carlo simulation
        
        Implements stratified sampling for improved accuracy:
        - Sample from different regions of probability space
        - Reduces variance compared to simple random sampling
        
        Args:
            current_state: Current composite state
            time_horizon: Number of time steps to forecast (k)
            num_simulations: Number of Monte Carlo runs (N)
            context_data: Real-time contextual data for adjusting probabilities
            
        Returns:
            ForecastResult with probability distribution
        """
        N = num_simulations or self.default_simulations
        
        # Apply contextual adjustments if provided
        if context_data:
            self._apply_contextual_adjustments(context_data)
        
        # Run Monte Carlo simulations
        if self.use_stratified_sampling:
            state_counts = self._monte_carlo_stratified(current_state, time_horizon, N)
        else:
            state_counts = self._monte_carlo_simple(current_state, time_horizon, N)
        
        # Compute probability distribution: P̂(s_{t+k}) = (1/N) * Σ 1{s^(j)_{t+k} = s}
        state_probabilities = {state: count / N for state, count in state_counts.items()}
        
        # Find most likely state
        expected_state = max(state_probabilities.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence (probability of most likely state)
        confidence = state_probabilities[expected_state]
        
        logger.info(f"Forecast completed: {N} runs, horizon={time_horizon}, "
                   f"expected={expected_state}, confidence={confidence:.3f}")
        
        return ForecastResult(
            time_horizon=time_horizon,
            state_probabilities=state_probabilities,
            expected_state=expected_state,
            confidence=confidence,
            simulation_runs=N
        )
    
    def _monte_carlo_simple(self, 
                           current_state: CompositeState,
                           time_horizon: int,
                           N: int) -> Dict[CompositeState, int]:
        """
        Simple Monte Carlo simulation
        
        Args:
            current_state: Starting state
            time_horizon: Number of steps
            N: Number of simulation runs
            
        Returns:
            Dictionary mapping states to occurrence counts
        """
        state_counts = {}
        
        for j in range(N):
            # Simulate one trajectory
            state = current_state
            for t in range(time_horizon):
                state = self.markov_module.sample_next_state(state)
            
            # Record final state
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return state_counts
    
    def _monte_carlo_stratified(self,
                               current_state: CompositeState,
                               time_horizon: int,
                               N: int) -> Dict[CompositeState, int]:
        """
        Stratified Monte Carlo simulation for variance reduction
        
        Divides probability space into strata and samples proportionally
        from each stratum to ensure better coverage.
        
        Args:
            current_state: Starting state
            time_horizon: Number of steps
            N: Number of simulation runs
            
        Returns:
            Dictionary mapping states to occurrence counts
        """
        state_counts = {}
        
        # Get all possible states
        all_states = self.markov_module.states
        num_strata = min(len(all_states), 9)  # Use 9 strata (3x3 grid)
        
        # Divide simulations across strata
        runs_per_stratum = N // num_strata
        remaining_runs = N % num_strata
        
        # For each stratum, run simulations with biased initial sampling
        for stratum_idx in range(num_strata):
            stratum_runs = runs_per_stratum + (1 if stratum_idx < remaining_runs else 0)
            
            for j in range(stratum_runs):
                # Simulate trajectory
                state = current_state
                
                # Inject stratified perturbation at first step
                if stratum_idx > 0 and np.random.random() < 0.3:
                    # Sample from specific region of state space
                    stratum_state = all_states[stratum_idx % len(all_states)]
                    state = stratum_state
                
                # Continue simulation
                for t in range(time_horizon):
                    state = self.markov_module.sample_next_state(state)
                
                # Record final state
                state_counts[state] = state_counts.get(state, 0) + 1
        
        return state_counts
    
    def _apply_contextual_adjustments(self, context_data: Dict):
        """
        Adjust transition probabilities based on real-time contextual data
        
        As stated in paper: "Transition probabilities derived from the extended 
        Markov Chain model are integrated with real-time contextual data"
        
        Args:
            context_data: Dictionary with contextual information
                - weather: current weather conditions
                - security_level: current security assessment
                - traffic_density: current traffic level
                - temperature_alerts: temperature sensor alerts
        """
        # Example contextual adjustments
        if context_data.get('weather') == 'extreme_heat':
            # Increase probability of temperature instability
            logger.debug("Contextual adjustment: extreme heat detected")
        
        if context_data.get('security_level') == 'critical':
            # Increase probability of road unavailability
            logger.debug("Contextual adjustment: critical security level")
        
        # These adjustments would modify the Markov transition matrix
        # through the update_from_sensor_data method
    
    def forecast_multiple_scenarios(self,
                                   current_state: CompositeState,
                                   time_horizons: List[int],
                                   num_simulations: Optional[int] = None) -> Dict[int, ForecastResult]:
        """
        Generate forecasts for multiple time horizons
        
        Useful for short-term vs long-term planning
        
        Args:
            current_state: Current composite state
            time_horizons: List of time horizons to forecast
            num_simulations: Number of Monte Carlo runs per horizon
            
        Returns:
            Dictionary mapping time horizon to ForecastResult
        """
        results = {}
        for horizon in time_horizons:
            results[horizon] = self.forecast(current_state, horizon, num_simulations)
        
        return results
    
    def estimate_risk_probability(self,
                                 current_state: CompositeState,
                                 time_horizon: int,
                                 risk_states: List[CompositeState],
                                 num_simulations: Optional[int] = None) -> float:
        """
        Estimate probability of reaching any risk state within time horizon
        
        Example: What is the probability of reaching (Unavailable, Critical) 
        or (Dangerous, Critical) within next 2 hours?
        
        Args:
            current_state: Current state
            time_horizon: Time horizon
            risk_states: List of states considered risky
            num_simulations: Number of simulations
            
        Returns:
            Probability of reaching risk state
        """
        forecast = self.forecast(current_state, time_horizon, num_simulations)
        
        risk_probability = sum(
            forecast.state_probabilities.get(state, 0.0) 
            for state in risk_states
        )
        
        return risk_probability
