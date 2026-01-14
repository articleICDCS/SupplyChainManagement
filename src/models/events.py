"""
Modèle de données pour les événements critiques
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Types d'événements critiques"""
    REFRIGERATION_FAILURE = "refrigeration_failure"
    POWER_OUTAGE = "power_outage"
    ROAD_BLOCKAGE = "road_blockage"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    EXTREME_WEATHER = "extreme_weather"
    SECURITY_THREAT = "security_threat"
    TRAFFIC_DELAY = "traffic_delay"
    FUEL_SHORTAGE = "fuel_shortage"


class EventSeverity(Enum):
    """Niveaux de sévérité"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass
class CriticalEvent:
    """
    Modèle d'un événement critique dans la chaîne du froid
    """
    id: str
    event_type: EventType
    severity: EventSeverity
    timestamp: datetime
    location: tuple  # (latitude, longitude)
    
    # Détails de l'événement
    description: str
    duration: Optional[float] = None  # heures (None si durée inconnue)
    affected_area_radius: float = 0.0  # km
    
    # Impact
    temperature_impact: float = 0.0  # Changement de température (°C)
    delay_impact: float = 0.0  # Délai causé (heures)
    cost_impact: float = 0.0  # Impact financier
    
    # Entités affectées
    affected_vehicles: list = None
    affected_medicines: list = None
    affected_routes: list = None
    
    # Résolution
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    mitigation_actions: list = None
    
    # Métadonnées
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.affected_vehicles is None:
            self.affected_vehicles = []
        if self.affected_medicines is None:
            self.affected_medicines = []
        if self.affected_routes is None:
            self.affected_routes = []
        if self.mitigation_actions is None:
            self.mitigation_actions = []
        if self.metadata is None:
            self.metadata = {}
    
    def get_duration(self) -> float:
        """Retourne la durée de l'événement"""
        if self.resolved and self.resolution_time:
            return (self.resolution_time - self.timestamp).total_seconds() / 3600
        elif self.duration:
            return self.duration
        else:
            return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    def get_impact_score(self) -> float:
        """
        Calcule un score d'impact de l'événement (0-1)
        """
        # Poids des différents facteurs
        severity_weight = 0.3
        duration_weight = 0.2
        affected_entities_weight = 0.3
        cost_weight = 0.2
        
        # Score de sévérité
        severity_score = self.severity.value / 5.0
        
        # Score de durée (normalisé à 24h)
        duration_score = min(self.get_duration() / 24.0, 1.0)
        
        # Score des entités affectées (normalisé à 10 entités)
        affected_count = len(self.affected_vehicles) + len(self.affected_medicines)
        affected_score = min(affected_count / 10.0, 1.0)
        
        # Score de coût (normalisé à 100000)
        cost_score = min(self.cost_impact / 100000.0, 1.0)
        
        return (
            severity_score * severity_weight +
            duration_score * duration_weight +
            affected_score * affected_entities_weight +
            cost_score * cost_weight
        )
    
    def to_dict(self) -> dict:
        """Convertit l'événement en dictionnaire"""
        return {
            'id': self.id,
            'type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'description': self.description,
            'duration': self.get_duration(),
            'resolved': self.resolved,
            'impact_score': self.get_impact_score(),
            'affected_vehicles_count': len(self.affected_vehicles),
            'affected_medicines_count': len(self.affected_medicines),
            'temperature_impact': self.temperature_impact,
            'delay_impact': self.delay_impact,
            'cost_impact': self.cost_impact
        }


@dataclass
class EnvironmentConditions:
    """
    Conditions environnementales affectant la chaîne du froid
    """
    timestamp: datetime
    location: tuple  # (latitude, longitude)
    
    # Conditions météorologiques
    ambient_temperature: float  # °C
    humidity: float  # %
    wind_speed: float  # km/h
    weather_condition: str  # "normal", "rain", "snow", "storm", etc.
    
    # Conditions d'infrastructure
    road_condition: str  # "good", "degraded", "blocked"
    power_availability: float  # 0-1 (pourcentage)
    communication_quality: float  # 0-1
    
    # Conditions de sécurité
    security_level: str  # "safe", "caution", "danger", "war_zone"
    conflict_intensity: float  # 0-1
    
    def is_extreme_weather(self) -> bool:
        """Vérifie si les conditions météo sont extrêmes"""
        return (
            self.ambient_temperature > 35 or
            self.ambient_temperature < -10 or
            self.weather_condition in ["storm", "extreme_heat", "extreme_cold"] or
            self.wind_speed > 50
        )
    
    def is_high_risk_zone(self) -> bool:
        """Vérifie si la zone est à haut risque"""
        return (
            self.security_level in ["danger", "war_zone"] or
            self.conflict_intensity > 0.7 or
            self.road_condition == "blocked"
        )
    
    def get_risk_multiplier(self) -> float:
        """
        Calcule un multiplicateur de risque basé sur les conditions (1.0 = normal)
        """
        multiplier = 1.0
        
        # Impact météo
        if self.is_extreme_weather():
            multiplier *= 1.5
        
        # Impact sécurité
        if self.security_level == "war_zone":
            multiplier *= 2.0
        elif self.security_level == "danger":
            multiplier *= 1.5
        elif self.security_level == "caution":
            multiplier *= 1.2
        
        # Impact infrastructure
        if self.power_availability < 0.5:
            multiplier *= 1.3
        if self.road_condition == "degraded":
            multiplier *= 1.2
        elif self.road_condition == "blocked":
            multiplier *= 3.0
        
        return multiplier
    
    def to_dict(self) -> dict:
        """Convertit les conditions en dictionnaire"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'ambient_temperature': self.ambient_temperature,
            'humidity': self.humidity,
            'weather_condition': self.weather_condition,
            'road_condition': self.road_condition,
            'security_level': self.security_level,
            'is_extreme': self.is_extreme_weather(),
            'is_high_risk': self.is_high_risk_zone(),
            'risk_multiplier': self.get_risk_multiplier()
        }
