"""
Modèle de données pour les véhicules de transport réfrigéré
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
from enum import Enum


class VehicleStatus(Enum):
    """États possibles d'un véhicule"""
    IDLE = "idle"
    IN_TRANSIT = "in_transit"
    LOADING = "loading"
    UNLOADING = "unloading"
    MAINTENANCE = "maintenance"
    BREAKDOWN = "breakdown"


class RefrigerationStatus(Enum):
    """États du système de réfrigération"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    EMERGENCY = "emergency"


@dataclass
class Vehicle:
    """
    Modèle d'un véhicule de transport réfrigéré
    """
    id: str
    vehicle_type: str
    capacity: float  # kg
    temperature_range: Tuple[float, float]  # (min, max) en °C
    autonomy: float  # km
    avg_speed: float  # km/h
    refrigeration_power: float  # W
    
    # État actuel
    current_status: VehicleStatus = VehicleStatus.IDLE
    current_location: Tuple[float, float] = (0.0, 0.0)  # (latitude, longitude)
    current_temperature: float = 4.0  # °C
    current_load: float = 0.0  # kg
    fuel_level: float = 1.0  # 0-1 (pourcentage)
    
    # Historique et maintenance
    age: float = 0.0  # années
    last_maintenance: Optional[datetime] = None
    total_distance: float = 0.0  # km
    breakdown_count: int = 0
    
    # Système de réfrigération
    refrigeration_status: RefrigerationStatus = RefrigerationStatus.OPERATIONAL
    refrigeration_efficiency: float = 1.0  # 0-1
    battery_level: float = 1.0  # 0-1 (pour système électrique)
    
    # Route et mission actuelle
    current_route: Optional[List[Tuple[float, float]]] = None
    destination: Optional[Tuple[float, float]] = None
    estimated_arrival: Optional[datetime] = None
    
    def get_remaining_capacity(self) -> float:
        """Retourne la capacité restante du véhicule"""
        return self.capacity - self.current_load
    
    def is_temperature_compliant(self) -> bool:
        """Vérifie si la température est dans la plage acceptable"""
        return self.temperature_range[0] <= self.current_temperature <= self.temperature_range[1]
    
    def get_risk_score(self) -> float:
        """
        Calcule un score de risque basé sur l'état du véhicule
        Retourne un score entre 0 (faible risque) et 1 (risque élevé)
        """
        risk_factors = []
        
        # Facteur d'âge
        risk_factors.append(min(self.age / 15.0, 1.0) * 0.2)
        
        # Facteur de maintenance
        if self.last_maintenance:
            days_since_maintenance = (datetime.now() - self.last_maintenance).days
            risk_factors.append(min(days_since_maintenance / 180.0, 1.0) * 0.15)
        else:
            risk_factors.append(0.15)
        
        # Facteur d'historique de pannes
        risk_factors.append(min(self.breakdown_count / 10.0, 1.0) * 0.25)
        
        # Facteur de réfrigération
        if self.refrigeration_status == RefrigerationStatus.FAILED:
            risk_factors.append(1.0 * 0.3)
        elif self.refrigeration_status == RefrigerationStatus.DEGRADED:
            risk_factors.append(0.5 * 0.3)
        else:
            risk_factors.append((1 - self.refrigeration_efficiency) * 0.3)
        
        # Facteur de carburant/batterie
        risk_factors.append((1 - min(self.fuel_level, self.battery_level)) * 0.1)
        
        return sum(risk_factors)
    
    def to_dict(self) -> dict:
        """Convertit le véhicule en dictionnaire"""
        return {
            'id': self.id,
            'type': self.vehicle_type,
            'status': self.current_status.value,
            'location': self.current_location,
            'temperature': self.current_temperature,
            'load': self.current_load,
            'capacity': self.capacity,
            'risk_score': self.get_risk_score(),
            'refrigeration_status': self.refrigeration_status.value
        }


@dataclass
class VehicleFleet:
    """Gestion d'une flotte de véhicules"""
    vehicles: List[Vehicle] = field(default_factory=list)
    
    def add_vehicle(self, vehicle: Vehicle):
        """Ajoute un véhicule à la flotte"""
        self.vehicles.append(vehicle)
    
    def get_available_vehicles(self) -> List[Vehicle]:
        """Retourne les véhicules disponibles"""
        return [v for v in self.vehicles if v.current_status == VehicleStatus.IDLE]
    
    def get_vehicles_by_type(self, vehicle_type: str) -> List[Vehicle]:
        """Retourne les véhicules d'un type donné"""
        return [v for v in self.vehicles if v.vehicle_type == vehicle_type]
    
    def get_high_risk_vehicles(self, threshold: float = 0.7) -> List[Vehicle]:
        """Retourne les véhicules à haut risque"""
        return [v for v in self.vehicles if v.get_risk_score() > threshold]
    
    def get_fleet_statistics(self) -> dict:
        """Retourne des statistiques sur la flotte"""
        if not self.vehicles:
            return {}
        
        return {
            'total_vehicles': len(self.vehicles),
            'available': len(self.get_available_vehicles()),
            'in_transit': len([v for v in self.vehicles if v.current_status == VehicleStatus.IN_TRANSIT]),
            'maintenance': len([v for v in self.vehicles if v.current_status == VehicleStatus.MAINTENANCE]),
            'breakdown': len([v for v in self.vehicles if v.current_status == VehicleStatus.BREAKDOWN]),
            'average_risk_score': sum(v.get_risk_score() for v in self.vehicles) / len(self.vehicles),
            'high_risk_count': len(self.get_high_risk_vehicles())
        }
