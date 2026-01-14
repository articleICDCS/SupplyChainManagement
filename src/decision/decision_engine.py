"""
Moteur de décision pour la chaîne du froid
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.vehicle import Vehicle, VehicleFleet
from ..models.medicine import Medicine, MedicineInventory
from ..models.events import EnvironmentConditions
from ..probabilistic.risk_models import BayesianRiskModel, RiskAssessment


class DecisionType(Enum):
    """Types de décisions"""
    PROCEED = "proceed"  # Continuer comme prévu
    DELAY = "delay"  # Reporter le transport
    REROUTE = "reroute"  # Changer de route
    CHANGE_VEHICLE = "change_vehicle"  # Changer de véhicule
    EMERGENCY_ACTION = "emergency_action"  # Action d'urgence
    ABORT = "abort"  # Annuler le transport


@dataclass
class Decision:
    """Représente une décision"""
    decision_type: DecisionType
    confidence: float  # 0-1
    reasoning: str
    risk_assessment: RiskAssessment
    alternative_actions: List[DecisionType]
    estimated_cost: float
    estimated_benefit: float
    priority: int  # 1-5


@dataclass
class TransportPlan:
    """Plan de transport optimisé"""
    vehicle: Vehicle
    medicines: List[Medicine]
    route: List[Tuple[float, float]]
    estimated_duration: float
    risk_score: float
    decisions: List[Decision]
    contingency_plans: List[str]


class DecisionEngine:
    """
    Moteur de décision probabiliste pour la gestion de la chaîne du froid
    """
    
    def __init__(
        self,
        fleet: VehicleFleet,
        inventory: MedicineInventory,
        risk_model: Optional[BayesianRiskModel] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialise le moteur de décision
        
        Args:
            fleet: Flotte de véhicules
            inventory: Inventaire de médicaments
            risk_model: Modèle de risque probabiliste
            config: Configuration des seuils et paramètres
        """
        self.fleet = fleet
        self.inventory = inventory
        self.risk_model = risk_model or BayesianRiskModel()
        self.config = config or self._default_config()
        
        self.decision_history: List[Decision] = []
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            },
            'optimization_weights': {
                'risk': 0.4,
                'time': 0.3,
                'cost': 0.2,
                'reliability': 0.1
            },
            'decision_thresholds': {
                'proceed': 0.5,
                'delay': 0.7,
                'abort': 0.9
            }
        }
    
    def evaluate_transport_feasibility(
        self,
        vehicle: Vehicle,
        medicine: Medicine,
        conditions: Optional[EnvironmentConditions] = None,
        transit_time: float = 4.0
    ) -> Decision:
        """
        Évalue la faisabilité d'un transport
        
        Args:
            vehicle: Véhicule proposé
            medicine: Médicament à transporter
            conditions: Conditions environnementales
            transit_time: Temps de transit estimé (heures)
            
        Returns:
            Décision recommandée
        """
        # Évaluation des risques
        vehicle_risk = self.risk_model.assess_vehicle_risk(vehicle, conditions)
        medicine_risk = self.risk_model.assess_medicine_risk(medicine, transit_time, vehicle)
        
        # Score de risque combiné
        combined_risk_score = (vehicle_risk.risk_score + medicine_risk.risk_score) / 2
        combined_confidence = (vehicle_risk.confidence + medicine_risk.confidence) / 2
        
        # Analyse de compatibilité véhicule-médicament
        temp_compatible = (
            vehicle.temperature_range[0] <= medicine.optimal_temp_range[0] and
            vehicle.temperature_range[1] >= medicine.optimal_temp_range[1]
        )
        
        if not temp_compatible:
            return Decision(
                decision_type=DecisionType.CHANGE_VEHICLE,
                confidence=0.95,
                reasoning="Incompatibilité de température entre véhicule et médicament",
                risk_assessment=vehicle_risk,
                alternative_actions=[DecisionType.ABORT],
                estimated_cost=1000.0,
                estimated_benefit=medicine.get_total_value(),
                priority=5
            )
        
        # Vérification de la capacité
        if vehicle.get_remaining_capacity() < medicine.quantity:
            return Decision(
                decision_type=DecisionType.CHANGE_VEHICLE,
                confidence=0.90,
                reasoning="Capacité du véhicule insuffisante",
                risk_assessment=vehicle_risk,
                alternative_actions=[DecisionType.ABORT],
                estimated_cost=500.0,
                estimated_benefit=medicine.get_total_value(),
                priority=4
            )
        
        # Vérification du temps disponible
        if medicine.get_remaining_safe_time() < transit_time:
            return Decision(
                decision_type=DecisionType.EMERGENCY_ACTION,
                confidence=0.85,
                reasoning="Temps disponible insuffisant - Transport d'urgence requis",
                risk_assessment=medicine_risk,
                alternative_actions=[DecisionType.ABORT],
                estimated_cost=5000.0,
                estimated_benefit=medicine.get_total_value(),
                priority=5
            )
        
        # Décision basée sur le risque combiné
        thresholds = self.config['risk_thresholds']
        
        if combined_risk_score < thresholds['low']:
            decision_type = DecisionType.PROCEED
            reasoning = "Risque faible - Transport approuvé"
            alternatives = []
            priority = 2
            
        elif combined_risk_score < thresholds['medium']:
            decision_type = DecisionType.PROCEED
            reasoning = "Risque modéré - Transport approuvé avec surveillance"
            alternatives = [DecisionType.DELAY]
            priority = 3
            
        elif combined_risk_score < thresholds['high']:
            # Analyse plus approfondie nécessaire
            if vehicle_risk.risk_score > medicine_risk.risk_score:
                decision_type = DecisionType.CHANGE_VEHICLE
                reasoning = "Risque véhicule élevé - Changement de véhicule recommandé"
                alternatives = [DecisionType.DELAY, DecisionType.PROCEED]
            else:
                decision_type = DecisionType.DELAY
                reasoning = "Risque élevé - Reporter jusqu'à amélioration des conditions"
                alternatives = [DecisionType.PROCEED, DecisionType.ABORT]
            priority = 4
            
        else:  # Risque critique
            if conditions and conditions.is_high_risk_zone():
                decision_type = DecisionType.ABORT
                reasoning = "Risque critique - Zone dangereuse - Transport déconseillé"
                alternatives = [DecisionType.EMERGENCY_ACTION]
            else:
                decision_type = DecisionType.EMERGENCY_ACTION
                reasoning = "Risque critique - Mesures d'urgence requises"
                alternatives = [DecisionType.ABORT]
            priority = 5
        
        # Calcul coût/bénéfice
        estimated_cost = self._estimate_transport_cost(
            vehicle,
            transit_time,
            decision_type
        )
        estimated_benefit = medicine.get_total_value() * (1 - combined_risk_score)
        
        return Decision(
            decision_type=decision_type,
            confidence=combined_confidence,
            reasoning=reasoning,
            risk_assessment=vehicle_risk,
            alternative_actions=alternatives,
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            priority=priority
        )
    
    def optimize_vehicle_assignment(
        self,
        medicines: List[Medicine],
        conditions: Optional[EnvironmentConditions] = None
    ) -> List[TransportPlan]:
        """
        Optimise l'affectation des véhicules aux médicaments
        
        Args:
            medicines: Liste de médicaments à transporter
            conditions: Conditions environnementales
            
        Returns:
            Liste de plans de transport optimisés
        """
        transport_plans = []
        available_vehicles = self.fleet.get_available_vehicles()
        
        if not available_vehicles:
            return []
        
        # Trier les médicaments par priorité et risque
        sorted_medicines = sorted(
            medicines,
            key=lambda m: (m.priority_level, -m.get_risk_score()),
            reverse=True
        )
        
        for medicine in sorted_medicines:
            best_plan = None
            best_score = -np.inf
            
            for vehicle in available_vehicles:
                # Évaluer cette combinaison
                decision = self.evaluate_transport_feasibility(
                    vehicle,
                    medicine,
                    conditions
                )
                
                if decision.decision_type in [DecisionType.PROCEED, DecisionType.DELAY]:
                    # Score combiné basé sur les poids de configuration
                    weights = self.config['optimization_weights']
                    score = (
                        (1 - decision.risk_assessment.risk_score) * weights['risk'] +
                        decision.confidence * weights['reliability'] +
                        (decision.estimated_benefit / max(decision.estimated_cost, 1)) * weights['cost']
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_plan = TransportPlan(
                            vehicle=vehicle,
                            medicines=[medicine],
                            route=[vehicle.current_location, medicine.destination or (0, 0)],
                            estimated_duration=4.0,  # À calculer proprement
                            risk_score=decision.risk_assessment.risk_score,
                            decisions=[decision],
                            contingency_plans=decision.risk_assessment.recommendations
                        )
            
            if best_plan:
                transport_plans.append(best_plan)
                # Retirer le véhicule des disponibles
                if best_plan.vehicle in available_vehicles:
                    available_vehicles.remove(best_plan.vehicle)
        
        return transport_plans
    
    def generate_contingency_plan(
        self,
        plan: TransportPlan,
        potential_events: List[str]
    ) -> Dict[str, List[str]]:
        """
        Génère des plans de contingence pour différents scénarios
        
        Args:
            plan: Plan de transport
            potential_events: Événements potentiels
            
        Returns:
            Dictionnaire d'actions par type d'événement
        """
        contingency_plans = {}
        
        for event_type in potential_events:
            actions = []
            
            if event_type == "refrigeration_failure":
                actions.extend([
                    "Activer le système de réfrigération de secours",
                    "Contacter le support technique immédiatement",
                    "Identifier l'entrepôt frigorifique le plus proche",
                    "Préparer le transfert vers un véhicule de remplacement"
                ])
            
            elif event_type == "road_blockage":
                actions.extend([
                    "Activer la route alternative pré-calculée",
                    "Estimer le délai supplémentaire",
                    "Vérifier l'autonomie du véhicule pour la nouvelle route",
                    "Notifier tous les points de livraison du retard"
                ])
            
            elif event_type == "extreme_weather":
                actions.extend([
                    "Évaluer si le véhicule peut supporter les conditions",
                    "Considérer un arrêt temporaire dans un lieu sûr",
                    "Augmenter la fréquence de monitoring de température",
                    "Préparer l'évacuation d'urgence si nécessaire"
                ])
            
            elif event_type == "temperature_excursion":
                actions.extend([
                    "Documenter précisément l'excursion (temps, température)",
                    "Évaluer l'impact sur chaque médicament",
                    "Contacter les autorités sanitaires si nécessaire",
                    "Décider de la poursuite ou de l'interruption du transport"
                ])
            
            contingency_plans[event_type] = actions
        
        return contingency_plans
    
    def _estimate_transport_cost(
        self,
        vehicle: Vehicle,
        transit_time: float,
        decision_type: DecisionType
    ) -> float:
        """Estime le coût du transport"""
        base_cost = 100.0  # Coût de base
        fuel_cost = transit_time * 50.0  # 50€/heure
        
        if decision_type == DecisionType.EMERGENCY_ACTION:
            multiplier = 3.0
        elif decision_type == DecisionType.CHANGE_VEHICLE:
            multiplier = 1.5
        elif decision_type == DecisionType.DELAY:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return (base_cost + fuel_cost) * multiplier
    
    def get_decision_summary(self) -> Dict:
        """Retourne un résumé des décisions prises"""
        if not self.decision_history:
            return {}
        
        decisions_by_type = {}
        for decision in self.decision_history:
            dtype = decision.decision_type.value
            if dtype not in decisions_by_type:
                decisions_by_type[dtype] = 0
            decisions_by_type[dtype] += 1
        
        total_cost = sum(d.estimated_cost for d in self.decision_history)
        total_benefit = sum(d.estimated_benefit for d in self.decision_history)
        avg_confidence = np.mean([d.confidence for d in self.decision_history])
        
        return {
            'total_decisions': len(self.decision_history),
            'decisions_by_type': decisions_by_type,
            'total_estimated_cost': total_cost,
            'total_estimated_benefit': total_benefit,
            'net_benefit': total_benefit - total_cost,
            'average_confidence': avg_confidence
        }
