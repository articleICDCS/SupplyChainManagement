"""
Simulateur d'événements critiques pour la chaîne du froid
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..models.events import CriticalEvent, EventType, EventSeverity, EnvironmentConditions
from ..models.vehicle import Vehicle, RefrigerationStatus, VehicleStatus
from ..models.medicine import Medicine, MedicineStatus

logger = logging.getLogger(__name__)


class CriticalEventSimulator:
    """
    Simule les événements critiques affectant la chaîne du froid
    """
    
    def __init__(self, config: Optional[Dict] = None, random_seed: Optional[int] = None):
        """
        Initialise le simulateur d'événements
        
        Args:
            config: Configuration des probabilités d'événements
            random_seed: Seed pour la reproductibilité
        """
        self.config = config or self._default_config()
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.active_events: List[CriticalEvent] = []
        self.event_history: List[CriticalEvent] = []
        self.event_counter = 0
    
    def _default_config(self) -> Dict:
        """Configuration par défaut des probabilités d'événements"""
        return {
            'refrigeration_failure': {
                'base_probability': 0.05,
                'factors': {
                    'vehicle_age': 0.02,
                    'extreme_weather': 0.15,
                    'war_zone': 0.25
                }
            },
            'road_blockage': {
                'base_probability': 0.10,
                'factors': {
                    'war_zone': 0.40,
                    'natural_disaster': 0.30
                }
            },
            'power_outage': {
                'base_probability': 0.08,
                'factors': {
                    'infrastructure_damage': 0.35,
                    'extreme_weather': 0.20
                }
            },
            'vehicle_breakdown': {
                'base_probability': 0.06,
                'factors': {
                    'vehicle_age': 0.03,
                    'poor_maintenance': 0.25
                }
            },
            'extreme_weather': {
                'base_probability': 0.12
            },
            'security_threat': {
                'base_probability': 0.03,
                'factors': {
                    'war_zone': 0.45,
                    'conflict_area': 0.25
                }
            }
        }
    
    def calculate_event_probability(
        self,
        event_type: EventType,
        vehicle: Optional[Vehicle] = None,
        conditions: Optional[EnvironmentConditions] = None
    ) -> float:
        """
        Calcule la probabilité d'occurrence d'un événement
        
        Args:
            event_type: Type d'événement
            vehicle: Véhicule concerné (si applicable)
            conditions: Conditions environnementales
            
        Returns:
            Probabilité entre 0 et 1
        """
        event_key = event_type.value
        if event_key not in self.config:
            return 0.0
        
        config = self.config[event_key]
        probability = config['base_probability']
        
        # Ajustements basés sur le véhicule
        if vehicle and 'factors' in config:
            if 'vehicle_age' in config['factors']:
                age_factor = min(vehicle.age / 10.0, 1.0) * config['factors']['vehicle_age']
                probability += age_factor
            
            if 'poor_maintenance' in config['factors']:
                if vehicle.last_maintenance:
                    days_since = (datetime.now() - vehicle.last_maintenance).days
                    maintenance_factor = min(days_since / 180.0, 1.0) * config['factors']['poor_maintenance']
                    probability += maintenance_factor
        
        # Ajustements basés sur les conditions environnementales
        if conditions and 'factors' in config:
            if conditions.is_extreme_weather() and 'extreme_weather' in config['factors']:
                probability += config['factors']['extreme_weather']
            
            if conditions.security_level == 'war_zone' and 'war_zone' in config['factors']:
                probability += config['factors']['war_zone']
            
            if conditions.power_availability < 0.3 and 'infrastructure_damage' in config['factors']:
                probability += config['factors']['infrastructure_damage']
        
        return min(probability, 1.0)
    
    def simulate_refrigeration_failure(
        self,
        vehicle: Vehicle,
        conditions: Optional[EnvironmentConditions] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[CriticalEvent]:
        """
        Simule une panne de réfrigération
        
        Args:
            vehicle: Véhicule concerné
            conditions: Conditions environnementales
            timestamp: Horodatage de l'événement
            
        Returns:
            CriticalEvent si l'événement se produit, None sinon
        """
        probability = self.calculate_event_probability(
            EventType.REFRIGERATION_FAILURE,
            vehicle,
            conditions
        )
        
        if np.random.random() < probability:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Déterminer la sévérité
            severity_roll = np.random.random()
            if severity_roll < 0.1:
                severity = EventSeverity.CATASTROPHIC
                temp_impact = 15.0
                duration = np.random.uniform(4, 12)
            elif severity_roll < 0.3:
                severity = EventSeverity.CRITICAL
                temp_impact = 10.0
                duration = np.random.uniform(2, 6)
            elif severity_roll < 0.6:
                severity = EventSeverity.HIGH
                temp_impact = 7.0
                duration = np.random.uniform(1, 4)
            else:
                severity = EventSeverity.MEDIUM
                temp_impact = 3.0
                duration = np.random.uniform(0.5, 2)
            
            self.event_counter += 1
            event = CriticalEvent(
                id=f"REF_FAIL_{self.event_counter}",
                event_type=EventType.REFRIGERATION_FAILURE,
                severity=severity,
                timestamp=timestamp,
                location=vehicle.current_location,
                description=f"Panne de réfrigération sur véhicule {vehicle.id}",
                duration=duration,
                temperature_impact=temp_impact,
                affected_vehicles=[vehicle.id],
                metadata={
                    'vehicle_age': vehicle.age,
                    'refrigeration_status': vehicle.refrigeration_status.value
                }
            )
            
            # Appliquer l'impact sur le véhicule
            vehicle.refrigeration_status = RefrigerationStatus.FAILED
            vehicle.breakdown_count += 1
            
            self.active_events.append(event)
            self.event_history.append(event)
            logger.warning(f"Panne de réfrigération simulée: {event.id}")
            
            return event
        
        return None
    
    def simulate_road_blockage(
        self,
        location: Tuple[float, float],
        conditions: Optional[EnvironmentConditions] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[CriticalEvent]:
        """
        Simule un blocage de route
        
        Args:
            location: Localisation du blocage
            conditions: Conditions environnementales
            timestamp: Horodatage
            
        Returns:
            CriticalEvent si l'événement se produit, None sinon
        """
        probability = self.calculate_event_probability(
            EventType.ROAD_BLOCKAGE,
            conditions=conditions
        )
        
        if np.random.random() < probability:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Sévérité et impact
            severity_roll = np.random.random()
            if conditions and conditions.security_level == 'war_zone':
                severity = EventSeverity.CRITICAL
                duration = np.random.uniform(12, 48)
                delay_impact = np.random.uniform(6, 24)
            elif severity_roll < 0.3:
                severity = EventSeverity.HIGH
                duration = np.random.uniform(4, 12)
                delay_impact = np.random.uniform(2, 8)
            else:
                severity = EventSeverity.MEDIUM
                duration = np.random.uniform(1, 4)
                delay_impact = np.random.uniform(0.5, 3)
            
            self.event_counter += 1
            event = CriticalEvent(
                id=f"ROAD_BLOCK_{self.event_counter}",
                event_type=EventType.ROAD_BLOCKAGE,
                severity=severity,
                timestamp=timestamp,
                location=location,
                description=f"Blocage de route à {location}",
                duration=duration,
                delay_impact=delay_impact,
                affected_area_radius=np.random.uniform(5, 50),
                metadata={
                    'road_condition': conditions.road_condition if conditions else 'unknown'
                }
            )
            
            self.active_events.append(event)
            self.event_history.append(event)
            logger.warning(f"Blocage de route simulé: {event.id}")
            
            return event
        
        return None
    
    def simulate_extreme_weather(
        self,
        location: Tuple[float, float],
        timestamp: Optional[datetime] = None
    ) -> Optional[CriticalEvent]:
        """
        Simule des conditions météorologiques extrêmes
        
        Args:
            location: Localisation
            timestamp: Horodatage
            
        Returns:
            CriticalEvent si l'événement se produit, None sinon
        """
        probability = self.config['extreme_weather']['base_probability']
        
        if np.random.random() < probability:
            if timestamp is None:
                timestamp = datetime.now()
            
            weather_types = ['extreme_heat', 'extreme_cold', 'storm', 'heavy_snow']
            weather_type = np.random.choice(weather_types)
            
            if weather_type == 'extreme_heat':
                temp_impact = np.random.uniform(10, 25)
                severity = EventSeverity.HIGH if temp_impact > 15 else EventSeverity.MEDIUM
            elif weather_type == 'extreme_cold':
                temp_impact = -np.random.uniform(10, 20)
                severity = EventSeverity.MEDIUM
            else:
                temp_impact = np.random.uniform(5, 15)
                severity = EventSeverity.HIGH
            
            duration = np.random.uniform(2, 12)
            
            self.event_counter += 1
            event = CriticalEvent(
                id=f"WEATHER_{self.event_counter}",
                event_type=EventType.EXTREME_WEATHER,
                severity=severity,
                timestamp=timestamp,
                location=location,
                description=f"Météo extrême: {weather_type}",
                duration=duration,
                temperature_impact=temp_impact,
                affected_area_radius=np.random.uniform(50, 200),
                metadata={'weather_type': weather_type}
            )
            
            self.active_events.append(event)
            self.event_history.append(event)
            logger.warning(f"Météo extrême simulée: {event.id}")
            
            return event
        
        return None
    
    def update_events(self, current_time: datetime) -> List[CriticalEvent]:
        """
        Met à jour l'état des événements actifs
        
        Args:
            current_time: Temps actuel de simulation
            
        Returns:
            Liste des événements résolus
        """
        resolved_events = []
        
        for event in self.active_events[:]:
            if event.duration:
                elapsed = (current_time - event.timestamp).total_seconds() / 3600
                if elapsed >= event.duration and not event.resolved:
                    event.resolved = True
                    event.resolution_time = current_time
                    self.active_events.remove(event)
                    resolved_events.append(event)
                    logger.info(f"Événement résolu: {event.id}")
        
        return resolved_events
    
    def get_active_events_at_location(
        self,
        location: Tuple[float, float],
        radius: float = 10.0
    ) -> List[CriticalEvent]:
        """
        Retourne les événements actifs à proximité d'une localisation
        
        Args:
            location: Localisation (lat, lon)
            radius: Rayon de recherche en km
            
        Returns:
            Liste des événements actifs dans la zone
        """
        nearby_events = []
        
        for event in self.active_events:
            # Calcul simple de distance (approximatif)
            lat_diff = abs(event.location[0] - location[0])
            lon_diff = abs(event.location[1] - location[1])
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Conversion deg -> km
            
            if distance <= radius or distance <= event.affected_area_radius:
                nearby_events.append(event)
        
        return nearby_events
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur les événements"""
        if not self.event_history:
            return {}
        
        events_by_type = {}
        events_by_severity = {}
        
        for event in self.event_history:
            # Par type
            event_type = event.event_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = 0
            events_by_type[event_type] += 1
            
            # Par sévérité
            severity = event.severity.name
            if severity not in events_by_severity:
                events_by_severity[severity] = 0
            events_by_severity[severity] += 1
        
        total_impact = sum(event.get_impact_score() for event in self.event_history)
        resolved_count = len([e for e in self.event_history if e.resolved])
        
        return {
            'total_events': len(self.event_history),
            'active_events': len(self.active_events),
            'resolved_events': resolved_count,
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'average_impact_score': total_impact / len(self.event_history),
            'total_cost_impact': sum(e.cost_impact for e in self.event_history)
        }
