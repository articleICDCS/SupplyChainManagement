"""
Moteur de simulation principal pour la chaîne du froid
"""
import simpy
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import logging

from ..models.vehicle import Vehicle, VehicleFleet, VehicleStatus, RefrigerationStatus
from ..models.medicine import Medicine, MedicineInventory
from ..models.events import EnvironmentConditions
from .critical_events import CriticalEventSimulator

logger = logging.getLogger(__name__)


class ColdChainSimulation:
    """
    Moteur de simulation principal pour la gestion de la chaîne du froid
    Utilise SimPy pour la simulation d'événements discrets
    """
    
    def __init__(
        self,
        fleet: VehicleFleet,
        inventory: MedicineInventory,
        duration: float = 24.0,  # heures
        time_step: float = 0.1,  # heures
        config: Optional[Dict] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialise la simulation
        
        Args:
            fleet: Flotte de véhicules
            inventory: Inventaire de médicaments
            duration: Durée de simulation en heures
            time_step: Pas de temps en heures
            config: Configuration de simulation
            random_seed: Seed pour reproductibilité
        """
        self.env = simpy.Environment()
        self.fleet = fleet
        self.inventory = inventory
        self.duration = duration
        self.time_step = time_step
        self.config = config or {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Simulateur d'événements critiques
        self.event_simulator = CriticalEventSimulator(
            config=self.config.get('critical_events') if self.config else None,
            random_seed=random_seed
        )
        
        # Historique de simulation
        self.simulation_log: List[Dict] = []
        self.start_time = datetime.now()
        
        # Callbacks
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Ajoute un callback à exécuter à chaque pas de temps"""
        self.callbacks.append(callback)
    
    def simulate_vehicle_operation(self, vehicle: Vehicle):
        """
        Simule le fonctionnement d'un véhicule
        
        Args:
            vehicle: Véhicule à simuler
        """
        while True:
            current_time = self.start_time + timedelta(hours=self.env.now)
            
            # Simulation de l'évolution de la température
            if vehicle.current_status == VehicleStatus.IN_TRANSIT:
                # Obtenir les conditions environnementales
                conditions = self._get_environmental_conditions(vehicle.current_location)
                
                # Vérifier les événements critiques
                events = self.event_simulator.get_active_events_at_location(
                    vehicle.current_location
                )
                
                # Calculer l'impact sur la température
                temp_delta = 0.0
                
                # Impact des conditions météo
                if conditions.is_extreme_weather():
                    if conditions.ambient_temperature > 35:
                        temp_delta += np.random.uniform(0.5, 2.0) * self.time_step
                    elif conditions.ambient_temperature < -10:
                        temp_delta -= np.random.uniform(0.3, 1.5) * self.time_step
                
                # Impact des événements
                for event in events:
                    if vehicle.id in event.affected_vehicles:
                        temp_delta += event.temperature_impact * self.time_step
                
                # Impact du système de réfrigération
                if vehicle.refrigeration_status == RefrigerationStatus.OPERATIONAL:
                    # Compenser les variations de température
                    compensation = -temp_delta * vehicle.refrigeration_efficiency
                    temp_delta += compensation
                elif vehicle.refrigeration_status == RefrigerationStatus.DEGRADED:
                    # Compensation partielle
                    compensation = -temp_delta * vehicle.refrigeration_efficiency * 0.5
                    temp_delta += compensation
                # Si FAILED, pas de compensation
                
                vehicle.current_temperature += temp_delta
                
                # Simuler les pannes de réfrigération
                self.event_simulator.simulate_refrigeration_failure(
                    vehicle,
                    conditions,
                    current_time
                )
                
                # Enregistrer l'exposition pour les médicaments à bord
                for medicine in self.inventory.medicines:
                    if medicine.current_location == vehicle.id:
                        medicine.current_temperature = vehicle.current_temperature
                        if not medicine.is_temperature_optimal():
                            medicine.record_temperature_exposure(
                                vehicle.current_temperature,
                                self.time_step,
                                current_time
                            )
            
            # Log de l'état
            self._log_state(vehicle, current_time)
            
            # Attendre le prochain pas de temps
            yield self.env.timeout(self.time_step)
    
    def simulate_environment(self):
        """Simule l'évolution des conditions environnementales"""
        while True:
            current_time = self.start_time + timedelta(hours=self.env.now)
            
            # Simuler des événements météo aléatoires
            if np.random.random() < 0.02:  # 2% de chance par pas de temps
                # Choisir une localisation aléatoire
                if self.fleet.vehicles:
                    vehicle = np.random.choice(self.fleet.vehicles)
                    self.event_simulator.simulate_extreme_weather(
                        vehicle.current_location,
                        current_time
                    )
            
            # Simuler des blocages de route
            if np.random.random() < 0.01:
                if self.fleet.vehicles:
                    vehicle = np.random.choice(self.fleet.vehicles)
                    conditions = self._get_environmental_conditions(vehicle.current_location)
                    self.event_simulator.simulate_road_blockage(
                        vehicle.current_location,
                        conditions,
                        current_time
                    )
            
            # Mettre à jour les événements actifs
            self.event_simulator.update_events(current_time)
            
            yield self.env.timeout(self.time_step)
    
    def _get_environmental_conditions(
        self,
        location: tuple
    ) -> EnvironmentConditions:
        """
        Génère ou récupère les conditions environnementales pour une localisation
        
        Args:
            location: Localisation (lat, lon)
            
        Returns:
            Conditions environnementales
        """
        # Pour la démonstration, génère des conditions aléatoires
        # Dans un système réel, ceci récupérerait les vraies données
        
        current_time = self.start_time + timedelta(hours=self.env.now)
        
        # Température ambiante avec variation circadienne
        base_temp = 20.0
        hour_of_day = current_time.hour
        circadian_variation = 10 * np.sin((hour_of_day - 6) * np.pi / 12)
        ambient_temp = base_temp + circadian_variation + np.random.normal(0, 3)
        
        weather_conditions = ["normal", "rain", "snow", "storm"]
        weather_probs = [0.7, 0.15, 0.1, 0.05]
        weather = np.random.choice(weather_conditions, p=weather_probs)
        
        security_levels = ["safe", "caution", "danger", "war_zone"]
        security_probs = [0.6, 0.25, 0.1, 0.05]
        security = np.random.choice(security_levels, p=security_probs)
        
        return EnvironmentConditions(
            timestamp=current_time,
            location=location,
            ambient_temperature=ambient_temp,
            humidity=np.random.uniform(30, 90),
            wind_speed=np.random.uniform(0, 50),
            weather_condition=weather,
            road_condition=np.random.choice(["good", "degraded", "blocked"], p=[0.8, 0.15, 0.05]),
            power_availability=np.random.uniform(0.5, 1.0),
            communication_quality=np.random.uniform(0.6, 1.0),
            security_level=security,
            conflict_intensity=np.random.uniform(0, 0.3) if security != "war_zone" else np.random.uniform(0.5, 1.0)
        )
    
    def _log_state(self, vehicle: Vehicle, current_time: datetime):
        """Enregistre l'état actuel dans le log"""
        log_entry = {
            'timestamp': current_time.isoformat(),
            'simulation_time': self.env.now,
            'vehicle_id': vehicle.id,
            'vehicle_status': vehicle.current_status.value,
            'vehicle_temperature': vehicle.current_temperature,
            'refrigeration_status': vehicle.refrigeration_status.value,
            'location': vehicle.current_location,
            'risk_score': vehicle.get_risk_score()
        }
        
        self.simulation_log.append(log_entry)
    
    def run(self):
        """Lance la simulation"""
        logger.info(f"Démarrage de la simulation - Durée: {self.duration}h")
        
        # Démarrer les processus de simulation pour chaque véhicule
        for vehicle in self.fleet.vehicles:
            self.env.process(self.simulate_vehicle_operation(vehicle))
        
        # Démarrer la simulation environnementale
        self.env.process(self.simulate_environment())
        
        # Lancer la simulation
        self.env.run(until=self.duration)
        
        logger.info("Simulation terminée")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Retourne les résultats de la simulation
        
        Returns:
            Dictionnaire contenant les résultats et statistiques
        """
        return {
            'simulation_log': self.simulation_log,
            'fleet_statistics': self.fleet.get_fleet_statistics(),
            'inventory_statistics': self.inventory.get_inventory_statistics(),
            'event_statistics': self.event_simulator.get_statistics(),
            'duration': self.duration,
            'total_log_entries': len(self.simulation_log)
        }


class ScenarioSimulation:
    """
    Classe pour simuler différents scénarios de crise
    """
    
    @staticmethod
    def scenario_normal_operation(
        fleet: VehicleFleet,
        inventory: MedicineInventory,
        duration: float = 24.0
    ) -> Dict:
        """Scénario d'opération normale"""
        config = {
            'critical_events': {
                'refrigeration_failure': {'base_probability': 0.02},
                'road_blockage': {'base_probability': 0.03},
                'extreme_weather': {'base_probability': 0.05}
            }
        }
        
        sim = ColdChainSimulation(fleet, inventory, duration, config=config)
        return sim.run()
    
    @staticmethod
    def scenario_war_zone(
        fleet: VehicleFleet,
        inventory: MedicineInventory,
        duration: float = 24.0
    ) -> Dict:
        """Scénario en zone de conflit"""
        config = {
            'critical_events': {
                'refrigeration_failure': {
                    'base_probability': 0.05,
                    'factors': {'war_zone': 0.35}
                },
                'road_blockage': {
                    'base_probability': 0.15,
                    'factors': {'war_zone': 0.50}
                },
                'power_outage': {
                    'base_probability': 0.20,
                    'factors': {'infrastructure_damage': 0.45}
                },
                'security_threat': {
                    'base_probability': 0.10,
                    'factors': {'war_zone': 0.60}
                }
            }
        }
        
        sim = ColdChainSimulation(fleet, inventory, duration, config=config)
        return sim.run()
    
    @staticmethod
    def scenario_pandemic(
        fleet: VehicleFleet,
        inventory: MedicineInventory,
        duration: float = 24.0
    ) -> Dict:
        """Scénario de pandémie (type COVID)"""
        config = {
            'critical_events': {
                'refrigeration_failure': {
                    'base_probability': 0.08,  # Véhicules surchargés
                    'factors': {'extreme_weather': 0.20}
                },
                'road_blockage': {
                    'base_probability': 0.12,  # Restrictions de circulation
                },
                'vehicle_breakdown': {
                    'base_probability': 0.10,  # Maintenance réduite
                }
            }
        }
        
        sim = ColdChainSimulation(fleet, inventory, duration, config=config)
        return sim.run()
