"""
Modèles probabilistes pour l'évaluation des risques
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from ..models.vehicle import Vehicle
from ..models.medicine import Medicine
from ..models.events import CriticalEvent, EnvironmentConditions


@dataclass
class RiskAssessment:
    """Résultat d'une évaluation de risque"""
    risk_score: float  # 0-1
    confidence: float  # 0-1
    risk_category: str  # "low", "medium", "high", "critical"
    contributing_factors: Dict[str, float]
    recommendations: List[str]
    probability_of_failure: float
    expected_loss: float


class BayesianRiskModel:
    """
    Modèle bayésien pour l'évaluation probabiliste des risques
    """
    
    def __init__(self):
        """Initialise le modèle probabiliste"""
        # Priors pour différents types de risques
        self.priors = {
            'refrigeration_failure': 0.05,
            'temperature_excursion': 0.15,
            'delivery_delay': 0.20,
            'medicine_compromise': 0.10
        }
        
        # Paramètres des distributions
        self.params = {
            'temperature_variance': 2.0,
            'failure_rate_shape': 2.0,
            'failure_rate_scale': 1.0
        }
    
    def update_prior(self, event_type: str, observed_frequency: float):
        """
        Met à jour les priors basés sur les observations
        
        Args:
            event_type: Type d'événement
            observed_frequency: Fréquence observée
        """
        if event_type in self.priors:
            # Mise à jour bayésienne simple
            self.priors[event_type] = (self.priors[event_type] + observed_frequency) / 2
    
    def calculate_posterior_probability(
        self,
        prior: float,
        likelihood: float,
        evidence: float = 1.0
    ) -> float:
        """
        Calcule la probabilité postérieure selon le théorème de Bayes
        
        P(H|E) = P(E|H) * P(H) / P(E)
        
        Args:
            prior: Probabilité a priori P(H)
            likelihood: Vraisemblance P(E|H)
            evidence: Probabilité de l'évidence P(E)
            
        Returns:
            Probabilité postérieure
        """
        if evidence == 0:
            return 0.0
        return (likelihood * prior) / evidence
    
    def assess_vehicle_risk(
        self,
        vehicle: Vehicle,
        conditions: Optional[EnvironmentConditions] = None,
        historical_failures: int = 0,
        total_trips: int = 1
    ) -> RiskAssessment:
        """
        Évalue le risque associé à un véhicule
        
        Args:
            vehicle: Véhicule à évaluer
            conditions: Conditions environnementales
            historical_failures: Nombre de pannes historiques
            total_trips: Nombre total de trajets
            
        Returns:
            Évaluation complète du risque
        """
        contributing_factors = {}
        
        # 1. Risque basé sur l'âge du véhicule
        age_risk = self._calculate_age_risk(vehicle.age)
        contributing_factors['vehicle_age'] = age_risk
        
        # 2. Risque basé sur la maintenance
        maintenance_risk = self._calculate_maintenance_risk(vehicle)
        contributing_factors['maintenance'] = maintenance_risk
        
        # 3. Risque basé sur l'historique
        if total_trips > 0:
            observed_failure_rate = historical_failures / total_trips
            # Mise à jour bayésienne
            prior = self.priors['refrigeration_failure']
            likelihood = observed_failure_rate
            evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
            posterior = self.calculate_posterior_probability(prior, likelihood, evidence)
            contributing_factors['historical_performance'] = posterior
        else:
            contributing_factors['historical_performance'] = self.priors['refrigeration_failure']
        
        # 4. Risque environnemental
        if conditions:
            env_risk = self._calculate_environmental_risk(conditions)
            contributing_factors['environment'] = env_risk
        else:
            contributing_factors['environment'] = 0.0
        
        # 5. Risque lié à l'état actuel de réfrigération
        refrigeration_risk = vehicle.get_risk_score()
        contributing_factors['refrigeration_status'] = refrigeration_risk
        
        # Agrégation des risques avec pondération
        weights = {
            'vehicle_age': 0.15,
            'maintenance': 0.20,
            'historical_performance': 0.25,
            'environment': 0.20,
            'refrigeration_status': 0.20
        }
        
        risk_score = sum(
            contributing_factors[k] * weights[k]
            for k in weights.keys()
        )
        
        # Calcul de la confiance (basé sur la quantité de données)
        confidence = min(0.5 + (total_trips / 100.0), 0.95)
        
        # Catégorisation du risque
        if risk_score < 0.3:
            risk_category = "low"
        elif risk_score < 0.5:
            risk_category = "medium"
        elif risk_score < 0.7:
            risk_category = "high"
        else:
            risk_category = "critical"
        
        # Probabilité de défaillance
        prob_failure = self._calculate_failure_probability(vehicle, conditions)
        
        # Perte attendue (valeur à risque)
        expected_loss = self._calculate_expected_loss(vehicle, prob_failure)
        
        # Recommandations
        recommendations = self._generate_recommendations(
            risk_category,
            contributing_factors,
            vehicle
        )
        
        return RiskAssessment(
            risk_score=risk_score,
            confidence=confidence,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            probability_of_failure=prob_failure,
            expected_loss=expected_loss
        )
    
    def assess_medicine_risk(
        self,
        medicine: Medicine,
        transit_time: float,  # heures
        vehicle: Optional[Vehicle] = None
    ) -> RiskAssessment:
        """
        Évalue le risque pour un médicament
        
        Args:
            medicine: Médicament à évaluer
            transit_time: Temps de transit estimé
            vehicle: Véhicule de transport
            
        Returns:
            Évaluation du risque
        """
        contributing_factors = {}
        
        # 1. Risque de température
        temp_risk = medicine.get_risk_score()
        contributing_factors['temperature'] = temp_risk
        
        # 2. Risque temporel
        remaining_time = medicine.get_remaining_safe_time()
        if remaining_time > 0:
            time_risk = 1.0 - (remaining_time / (remaining_time + transit_time))
        else:
            time_risk = 1.0
        contributing_factors['time_constraint'] = time_risk
        
        # 3. Risque du véhicule
        if vehicle:
            vehicle_risk = vehicle.get_risk_score()
            contributing_factors['vehicle'] = vehicle_risk
        else:
            contributing_factors['vehicle'] = 0.3
        
        # 4. Risque d'expiration
        if medicine.expiry_date:
            from datetime import datetime
            days_until_expiry = (medicine.expiry_date - datetime.now()).days
            expiry_risk = max(0, 1.0 - (days_until_expiry / 365.0))
        else:
            expiry_risk = 0.0
        contributing_factors['expiration'] = expiry_risk
        
        # Agrégation
        weights = {
            'temperature': 0.40,
            'time_constraint': 0.30,
            'vehicle': 0.20,
            'expiration': 0.10
        }
        
        risk_score = sum(
            contributing_factors[k] * weights[k]
            for k in weights.keys()
        )
        
        # Catégorisation
        if risk_score < 0.3:
            risk_category = "low"
        elif risk_score < 0.5:
            risk_category = "medium"
        elif risk_score < 0.7:
            risk_category = "high"
        else:
            risk_category = "critical"
        
        # Probabilité de compromission
        prob_compromise = self._calculate_compromise_probability(
            medicine,
            transit_time,
            vehicle
        )
        
        # Perte attendue
        expected_loss = medicine.get_total_value() * prob_compromise
        
        # Recommandations
        recommendations = self._generate_medicine_recommendations(
            risk_category,
            contributing_factors,
            medicine
        )
        
        return RiskAssessment(
            risk_score=risk_score,
            confidence=0.8,  # Haute confiance sur les contraintes physiques
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            probability_of_failure=prob_compromise,
            expected_loss=expected_loss
        )
    
    def _calculate_age_risk(self, age: float) -> float:
        """Calcule le risque basé sur l'âge (distribution de Weibull)"""
        # Distribution de Weibull pour modéliser la défaillance
        shape = self.params['failure_rate_shape']
        scale = self.params['failure_rate_scale']
        return min(stats.weibull_min.cdf(age, shape, scale=scale * 10), 1.0)
    
    def _calculate_maintenance_risk(self, vehicle: Vehicle) -> float:
        """Calcule le risque lié à la maintenance"""
        if vehicle.last_maintenance is None:
            return 0.8
        
        from datetime import datetime
        days_since = (datetime.now() - vehicle.last_maintenance).days
        
        # Risque croît exponentiellement après 90 jours
        risk = 1.0 - np.exp(-days_since / 90.0)
        return min(risk, 1.0)
    
    def _calculate_environmental_risk(self, conditions: EnvironmentConditions) -> float:
        """Calcule le risque environnemental"""
        risk_factors = []
        
        if conditions.is_extreme_weather():
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.2)
        
        if conditions.is_high_risk_zone():
            risk_factors.append(0.8)
        else:
            risk_factors.append(0.1)
        
        return np.mean(risk_factors)
    
    def _calculate_failure_probability(
        self,
        vehicle: Vehicle,
        conditions: Optional[EnvironmentConditions]
    ) -> float:
        """Calcule la probabilité de défaillance"""
        base_prob = self.priors['refrigeration_failure']
        
        # Ajustements
        age_multiplier = 1 + (vehicle.age / 10.0)
        breakdown_multiplier = 1 + (vehicle.breakdown_count * 0.1)
        
        if conditions:
            env_multiplier = conditions.get_risk_multiplier()
        else:
            env_multiplier = 1.0
        
        prob = base_prob * age_multiplier * breakdown_multiplier * env_multiplier
        return min(prob, 0.95)
    
    def _calculate_compromise_probability(
        self,
        medicine: Medicine,
        transit_time: float,
        vehicle: Optional[Vehicle]
    ) -> float:
        """Calcule la probabilité de compromission du médicament"""
        base_prob = self.priors['medicine_compromise']
        
        # Facteur temporel
        if medicine.get_remaining_safe_time() < transit_time:
            time_multiplier = 3.0
        else:
            time_multiplier = 1.0
        
        # Facteur véhicule
        if vehicle:
            vehicle_multiplier = 1 + vehicle.get_risk_score()
        else:
            vehicle_multiplier = 1.5
        
        prob = base_prob * time_multiplier * vehicle_multiplier
        return min(prob, 0.95)
    
    def _calculate_expected_loss(self, vehicle: Vehicle, prob_failure: float) -> float:
        """Calcule la perte attendue"""
        # Coût estimé d'une défaillance
        maintenance_cost = 5000
        downtime_cost = 2000 * 24  # Par jour
        
        return (maintenance_cost + downtime_cost) * prob_failure
    
    def _generate_recommendations(
        self,
        risk_category: str,
        factors: Dict[str, float],
        vehicle: Vehicle
    ) -> List[str]:
        """Génère des recommandations basées sur l'évaluation"""
        recommendations = []
        
        if risk_category in ["high", "critical"]:
            recommendations.append("⚠️ Inspection immédiate du véhicule recommandée")
        
        if factors.get('maintenance', 0) > 0.5:
            recommendations.append("🔧 Maintenance préventive nécessaire")
        
        if factors.get('refrigeration_status', 0) > 0.6:
            recommendations.append("❄️ Vérifier le système de réfrigération")
        
        if factors.get('environment', 0) > 0.6:
            recommendations.append("🌡️ Conditions environnementales défavorables - Considérer un report")
        
        if vehicle.fuel_level < 0.3:
            recommendations.append("⛽ Niveau de carburant faible")
        
        return recommendations
    
    def _generate_medicine_recommendations(
        self,
        risk_category: str,
        factors: Dict[str, float],
        medicine: Medicine
    ) -> List[str]:
        """Génère des recommandations pour les médicaments"""
        recommendations = []
        
        if risk_category == "critical":
            recommendations.append("🚨 Transport urgent requis avec surveillance maximale")
        
        if factors.get('temperature', 0) > 0.6:
            recommendations.append("🌡️ Température actuelle préoccupante - Action immédiate")
        
        if factors.get('time_constraint', 0) > 0.7:
            recommendations.append("⏰ Temps limite critique - Prioriser ce transport")
        
        if medicine.get_remaining_safe_time() < 2:
            recommendations.append("⚡ Moins de 2h restantes hors température optimale")
        
        if medicine.priority_level >= 4:
            recommendations.append("🔴 Médicament haute priorité")
        
        return recommendations
