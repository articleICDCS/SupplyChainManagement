"""
Modèle de données pour les médicaments thermosensibles
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum


class MedicineStatus(Enum):
    """États possibles d'un lot de médicaments"""
    SAFE = "safe"
    AT_RISK = "at_risk"
    COMPROMISED = "compromised"
    DAMAGED = "damaged"


class StorageCondition(Enum):
    """Conditions de stockage"""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class TemperatureExposure:
    """Enregistrement d'une exposition à une température hors limites"""
    timestamp: datetime
    temperature: float  # °C
    duration: float  # heures
    severity: float  # 0-1, calculé selon l'écart à la température optimale


@dataclass
class Medicine:
    """
    Modèle d'un médicament thermosensible
    """
    id: str
    name: str
    medicine_type: str
    batch_number: str
    quantity: float  # kg ou unités
    
    # Contraintes de température
    optimal_temp_range: Tuple[float, float]  # (min, max) en °C
    critical_temp_high: float  # Température maximale critique
    critical_temp_low: float  # Température minimale critique
    exposure_time_limit: float  # Temps max hors température (heures)
    
    # État actuel
    current_temperature: float = 4.0  # °C
    current_status: MedicineStatus = MedicineStatus.SAFE
    current_location: str = "warehouse"
    
    # Historique d'exposition
    temperature_history: List[TemperatureExposure] = field(default_factory=list)
    total_exposure_time: float = 0.0  # heures
    
    # Informations logistiques
    manufacturing_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    value_per_unit: float = 0.0  # Valeur économique
    priority_level: int = 1  # 1-5, 5 étant le plus urgent
    
    # Destination
    origin: Optional[str] = None
    destination: Optional[str] = None
    required_delivery_time: Optional[datetime] = None
    
    def is_temperature_optimal(self, temperature: Optional[float] = None) -> bool:
        """Vérifie si la température est dans la plage optimale"""
        temp = temperature if temperature is not None else self.current_temperature
        return self.optimal_temp_range[0] <= temp <= self.optimal_temp_range[1]
    
    def is_temperature_critical(self, temperature: Optional[float] = None) -> bool:
        """Vérifie si la température est critique"""
        temp = temperature if temperature is not None else self.current_temperature
        return temp > self.critical_temp_high or temp < self.critical_temp_low
    
    def record_temperature_exposure(self, temperature: float, duration: float, timestamp: Optional[datetime] = None):
        """Enregistre une exposition à une température"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calcul de la sévérité
        optimal_center = (self.optimal_temp_range[0] + self.optimal_temp_range[1]) / 2
        optimal_range = self.optimal_temp_range[1] - self.optimal_temp_range[0]
        deviation = abs(temperature - optimal_center)
        severity = min(deviation / (optimal_range * 2), 1.0)
        
        exposure = TemperatureExposure(
            timestamp=timestamp,
            temperature=temperature,
            duration=duration,
            severity=severity
        )
        
        self.temperature_history.append(exposure)
        
        if not self.is_temperature_optimal(temperature):
            self.total_exposure_time += duration
            self._update_status()
    
    def _update_status(self):
        """Met à jour le statut du médicament"""
        if self.is_temperature_critical():
            self.current_status = MedicineStatus.DAMAGED
        elif self.total_exposure_time > self.exposure_time_limit:
            self.current_status = MedicineStatus.COMPROMISED
        elif self.total_exposure_time > self.exposure_time_limit * 0.5:
            self.current_status = MedicineStatus.AT_RISK
        elif not self.is_temperature_optimal():
            self.current_status = MedicineStatus.AT_RISK
        else:
            self.current_status = MedicineStatus.SAFE
    
    def get_remaining_safe_time(self) -> float:
        """Retourne le temps restant avant compromission (heures)"""
        return max(0, self.exposure_time_limit - self.total_exposure_time)
    
    def get_risk_score(self) -> float:
        """
        Calcule un score de risque pour le médicament
        Retourne un score entre 0 (faible risque) et 1 (risque élevé)
        """
        risk_factors = []
        
        # Facteur de température actuelle
        if self.is_temperature_critical():
            risk_factors.append(1.0 * 0.4)
        elif not self.is_temperature_optimal():
            optimal_center = (self.optimal_temp_range[0] + self.optimal_temp_range[1]) / 2
            deviation = abs(self.current_temperature - optimal_center)
            risk_factors.append(min(deviation / 20.0, 1.0) * 0.4)
        else:
            risk_factors.append(0.0)
        
        # Facteur d'exposition cumulative
        exposure_ratio = self.total_exposure_time / self.exposure_time_limit if self.exposure_time_limit > 0 else 0
        risk_factors.append(min(exposure_ratio, 1.0) * 0.3)
        
        # Facteur d'expiration
        if self.expiry_date:
            days_until_expiry = (self.expiry_date - datetime.now()).days
            if days_until_expiry < 0:
                risk_factors.append(1.0 * 0.15)
            else:
                risk_factors.append(max(0, 1 - days_until_expiry / 365.0) * 0.15)
        else:
            risk_factors.append(0.0)
        
        # Facteur d'urgence de livraison
        if self.required_delivery_time:
            hours_until_delivery = (self.required_delivery_time - datetime.now()).total_seconds() / 3600
            if hours_until_delivery < 0:
                risk_factors.append(1.0 * 0.15)
            else:
                risk_factors.append(max(0, 1 - hours_until_delivery / 48.0) * 0.15)
        else:
            risk_factors.append(0.0)
        
        return sum(risk_factors)
    
    def get_total_value(self) -> float:
        """Calcule la valeur totale du lot"""
        return self.quantity * self.value_per_unit
    
    def to_dict(self) -> dict:
        """Convertit le médicament en dictionnaire"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.medicine_type,
            'quantity': self.quantity,
            'status': self.current_status.value,
            'current_temperature': self.current_temperature,
            'total_exposure_time': self.total_exposure_time,
            'remaining_safe_time': self.get_remaining_safe_time(),
            'risk_score': self.get_risk_score(),
            'value': self.get_total_value(),
            'priority': self.priority_level
        }


@dataclass
class MedicineInventory:
    """Gestion de l'inventaire des médicaments"""
    medicines: List[Medicine] = field(default_factory=list)
    
    def add_medicine(self, medicine: Medicine):
        """Ajoute un médicament à l'inventaire"""
        self.medicines.append(medicine)
    
    def get_compromised_medicines(self) -> List[Medicine]:
        """Retourne les médicaments compromis ou endommagés"""
        return [m for m in self.medicines if m.current_status in [MedicineStatus.COMPROMISED, MedicineStatus.DAMAGED]]
    
    def get_high_priority_medicines(self, min_priority: int = 4) -> List[Medicine]:
        """Retourne les médicaments haute priorité"""
        return [m for m in self.medicines if m.priority_level >= min_priority]
    
    def get_high_risk_medicines(self, threshold: float = 0.7) -> List[Medicine]:
        """Retourne les médicaments à haut risque"""
        return [m for m in self.medicines if m.get_risk_score() > threshold]
    
    def get_total_value(self) -> float:
        """Calcule la valeur totale de l'inventaire"""
        return sum(m.get_total_value() for m in self.medicines)
    
    def get_inventory_statistics(self) -> dict:
        """Retourne des statistiques sur l'inventaire"""
        if not self.medicines:
            return {}
        
        return {
            'total_medicines': len(self.medicines),
            'safe': len([m for m in self.medicines if m.current_status == MedicineStatus.SAFE]),
            'at_risk': len([m for m in self.medicines if m.current_status == MedicineStatus.AT_RISK]),
            'compromised': len([m for m in self.medicines if m.current_status == MedicineStatus.COMPROMISED]),
            'damaged': len([m for m in self.medicines if m.current_status == MedicineStatus.DAMAGED]),
            'total_value': self.get_total_value(),
            'average_risk_score': sum(m.get_risk_score() for m in self.medicines) / len(self.medicines),
            'high_priority_count': len(self.get_high_priority_medicines())
        }
