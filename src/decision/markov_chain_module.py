"""
Markov Chain Module
Implements Continuous-Time Markov Chain (CTMC) for modeling road availability 
and cold chain stability as described in Section B of the research proposal.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RoadCondition(Enum):
    """Road condition states"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DANGEROUS = "dangerous"


class TemperatureCondition(Enum):
    """Temperature condition states"""
    STABLE = "stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


@dataclass
class CompositeState:
    """
    Composite state combining road condition and temperature condition.
    Example: (Available, Stable), (Dangerous, Critical)
    """
    road_condition: RoadCondition
    temperature_condition: TemperatureCondition
    
    def __str__(self):
        return f"({self.road_condition.value}, {self.temperature_condition.value})"
    
    def __hash__(self):
        return hash((self.road_condition, self.temperature_condition))
    
    def __eq__(self, other):
        return (self.road_condition == other.road_condition and 
                self.temperature_condition == other.temperature_condition)


class MarkovChainModule:
    """
    Markov Chain Module for modeling route availability and cold chain stability.
    
    Implements equation from paper:
    P(s_{t+1} | s_t) = T[s_t, s_{t+1}]
    
    where states are composite: (road_condition, temperature_condition)
    """
    
    def __init__(self, bayesian_update: bool = True):
        """
        Initialize Markov Chain Module
        
        Args:
            bayesian_update: Enable Bayesian updating of transition probabilities
        """
        self.bayesian_update = bayesian_update
        
        # Generate all possible composite states
        self.states = self._generate_composite_states()
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}
        
        # Initialize transition matrix
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Prior observations for Bayesian updating
        self.transition_counts = np.ones_like(self.transition_matrix)
        
        logger.info(f"Markov Chain initialized with {len(self.states)} composite states")
    
    def _generate_composite_states(self) -> List[CompositeState]:
        """Generate all possible composite states (road × temperature)"""
        states = []
        for road_cond in RoadCondition:
            for temp_cond in TemperatureCondition:
                states.append(CompositeState(road_cond, temp_cond))
        return states
    
    def _initialize_transition_matrix(self) -> np.ndarray:
        """
        Initialize transition probability matrix T[s_t, s_{t+1}]
        
        Returns:
            Transition matrix of shape (num_states, num_states)
        """
        n = len(self.states)
        T = np.zeros((n, n))
        
        # Define realistic transition probabilities
        for i, state_from in enumerate(self.states):
            for j, state_to in enumerate(self.states):
                T[i, j] = self._default_transition_probability(state_from, state_to)
        
        # Normalize rows to ensure valid probability distribution
        T = T / T.sum(axis=1, keepdims=True)
        
        return T
    
    def _default_transition_probability(self, 
                                       state_from: CompositeState, 
                                       state_to: CompositeState) -> float:
        """
        Define default transition probabilities based on realistic assumptions
        
        Transitions are more likely between similar states
        """
        # Same state (persistence)
        if state_from == state_to:
            return 0.7
        
        # Road condition changes
        road_change = (state_from.road_condition != state_to.road_condition)
        
        # Temperature condition changes
        temp_change = (state_from.temperature_condition != state_to.temperature_condition)
        
        # Both change (rare)
        if road_change and temp_change:
            return 0.01
        
        # Only road changes
        if road_change:
            # Available -> Dangerous: moderate probability in war zones
            if (state_from.road_condition == RoadCondition.AVAILABLE and 
                state_to.road_condition == RoadCondition.DANGEROUS):
                return 0.10
            # Dangerous -> Unavailable: high probability
            elif (state_from.road_condition == RoadCondition.DANGEROUS and 
                  state_to.road_condition == RoadCondition.UNAVAILABLE):
                return 0.15
            # Other road transitions
            else:
                return 0.05
        
        # Only temperature changes
        if temp_change:
            # Stable -> Unstable: moderate probability
            if (state_from.temperature_condition == TemperatureCondition.STABLE and 
                state_to.temperature_condition == TemperatureCondition.UNSTABLE):
                return 0.12
            # Unstable -> Critical: high probability
            elif (state_from.temperature_condition == TemperatureCondition.UNSTABLE and 
                  state_to.temperature_condition == TemperatureCondition.CRITICAL):
                return 0.18
            # Other temperature transitions
            else:
                return 0.08
        
        return 0.01
    
    def get_transition_probability(self, 
                                   state_from: CompositeState, 
                                   state_to: CompositeState) -> float:
        """
        Get transition probability P(s_{t+1} | s_t)
        
        Args:
            state_from: Current state
            state_to: Next state
            
        Returns:
            Transition probability
        """
        i = self.state_to_idx[state_from]
        j = self.state_to_idx[state_to]
        return self.transition_matrix[i, j]
    
    def predict_next_state(self, current_state: CompositeState) -> CompositeState:
        """
        Predict most likely next state given current state
        
        Args:
            current_state: Current composite state
            
        Returns:
            Most likely next state
        """
        i = self.state_to_idx[current_state]
        probabilities = self.transition_matrix[i, :]
        next_idx = np.argmax(probabilities)
        return self.idx_to_state[next_idx]
    
    def sample_next_state(self, current_state: CompositeState) -> CompositeState:
        """
        Sample next state according to transition probabilities
        
        Args:
            current_state: Current composite state
            
        Returns:
            Sampled next state
        """
        i = self.state_to_idx[current_state]
        probabilities = self.transition_matrix[i, :]
        next_idx = np.random.choice(len(self.states), p=probabilities)
        return self.idx_to_state[next_idx]
    
    def update_transition_matrix_bayesian(self, 
                                         observed_transitions: List[Tuple[CompositeState, CompositeState]]):
        """
        Update transition matrix using Bayesian updating mechanism
        Based on real-time sensor data and observations
        
        Args:
            observed_transitions: List of (state_from, state_to) tuples
        """
        if not self.bayesian_update:
            return
        
        # Update transition counts
        for state_from, state_to in observed_transitions:
            i = self.state_to_idx[state_from]
            j = self.state_to_idx[state_to]
            self.transition_counts[i, j] += 1
        
        # Recompute transition matrix with Bayesian posterior
        # Using Dirichlet-Multinomial conjugate prior
        self.transition_matrix = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        
        logger.info(f"Transition matrix updated with {len(observed_transitions)} observations")
    
    def update_from_sensor_data(self, 
                               segment_id: str,
                               current_state: CompositeState,
                               temperature_reading: float,
                               road_status: str,
                               ambient_temp: float):
        """
        Update transition probabilities dynamically based on sensor data
        
        Example from paper: "if ambient temperature fluctuations are detected, 
        or directly to (Unavailable, Critical) if a sudden disruption occurs"
        
        Args:
            segment_id: Road segment identifier
            current_state: Current composite state
            temperature_reading: Current temperature reading (°C)
            road_status: Current road status
            ambient_temp: Ambient temperature (°C)
        """
        # Detect temperature fluctuations
        temp_fluctuation = abs(ambient_temp - 25.0) > 10.0  # 10°C deviation from normal
        
        # Infer likely transitions based on sensor data
        observed_transitions = []
        
        # Temperature instability detected
        if temp_fluctuation and current_state.temperature_condition == TemperatureCondition.STABLE:
            new_state = CompositeState(current_state.road_condition, TemperatureCondition.UNSTABLE)
            observed_transitions.append((current_state, new_state))
        
        # Road disruption detected
        if road_status == "blocked" and current_state.road_condition == RoadCondition.AVAILABLE:
            new_state = CompositeState(RoadCondition.UNAVAILABLE, current_state.temperature_condition)
            observed_transitions.append((current_state, new_state))
        
        # Critical temperature detected
        if temperature_reading < -80 or temperature_reading > 10:
            new_state = CompositeState(current_state.road_condition, TemperatureCondition.CRITICAL)
            observed_transitions.append((current_state, new_state))
        
        # Update transition matrix with observations
        if observed_transitions:
            self.update_transition_matrix_bayesian(observed_transitions)
    
    def get_state_distribution(self, initial_state: CompositeState, time_steps: int) -> Dict[CompositeState, float]:
        """
        Compute state distribution after k time steps
        
        Args:
            initial_state: Initial composite state
            time_steps: Number of time steps
            
        Returns:
            Dictionary mapping states to probabilities
        """
        # Initial distribution (all probability on initial state)
        dist = np.zeros(len(self.states))
        dist[self.state_to_idx[initial_state]] = 1.0
        
        # Evolve distribution: π(t+k) = π(t) * T^k
        T_power = np.linalg.matrix_power(self.transition_matrix, time_steps)
        final_dist = dist @ T_power
        
        # Convert to dictionary
        return {state: final_dist[idx] for state, idx in self.state_to_idx.items()}
