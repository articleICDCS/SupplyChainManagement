"""
Exemple d'utilisation complète du système de décision probabiliste
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
import logging

from src.models import (
    Vehicle, VehicleFleet, VehicleStatus, RefrigerationStatus,
    Medicine, MedicineInventory, MedicineStatus,
    EnvironmentConditions
)
from src.simulation import ColdChainSimulation, ScenarioSimulation
from src.decision import DecisionEngine
from src.probabilistic import BayesianRiskModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_fleet():
    """Crée une flotte de véhicules d'exemple"""
    fleet = VehicleFleet()
    
    # Véhicule 1: Camion réfrigéré standard
    vehicle1 = Vehicle(
        id="TRUCK_001",
        vehicle_type="refrigerated_truck",
        capacity=1000,
        temperature_range=(2, 8),
        autonomy=500,
        avg_speed=60,
        refrigeration_power=5000,
        current_location=(48.8566, 2.3522),  # Paris
        age=3.0,
        last_maintenance=datetime.now() - timedelta(days=45)
    )
    
    # Véhicule 2: Camion pour ultra-froid
    vehicle2 = Vehicle(
        id="TRUCK_002",
        vehicle_type="cold_box",
        capacity=500,
        temperature_range=(-80, -60),
        autonomy=400,
        avg_speed=55,
        refrigeration_power=8000,
        current_location=(48.8566, 2.3522),
        age=1.5,
        last_maintenance=datetime.now() - timedelta(days=20)
    )
    
    # Véhicule 3: Ancien véhicule avec risques
    vehicle3 = Vehicle(
        id="TRUCK_003",
        vehicle_type="refrigerated_truck",
        capacity=800,
        temperature_range=(2, 8),
        autonomy=450,
        avg_speed=50,
        refrigeration_power=4000,
        current_location=(48.8566, 2.3522),
        age=8.0,
        last_maintenance=datetime.now() - timedelta(days=150),
        breakdown_count=3
    )
    
    fleet.add_vehicle(vehicle1)
    fleet.add_vehicle(vehicle2)
    fleet.add_vehicle(vehicle3)
    
    return fleet


def create_sample_inventory():
    """Crée un inventaire de médicaments d'exemple"""
    inventory = MedicineInventory()
    
    # Vaccin COVID ultra-froid
    medicine1 = Medicine(
        id="MED_001",
        name="Vaccin COVID-19 mRNA",
        medicine_type="vaccine_covid",
        batch_number="BATCH_2024_001",
        quantity=200,
        optimal_temp_range=(-70, -60),
        critical_temp_high=-50,
        critical_temp_low=-80,
        exposure_time_limit=2,
        current_temperature=-65,
        value_per_unit=50,
        priority_level=5,
        manufacturing_date=datetime.now() - timedelta(days=30),
        expiry_date=datetime.now() + timedelta(days=180),
        destination="Hospital_A"
    )
    
    # Vaccin standard
    medicine2 = Medicine(
        id="MED_002",
        name="Vaccin Standard",
        medicine_type="vaccine_standard",
        batch_number="BATCH_2024_002",
        quantity=500,
        optimal_temp_range=(2, 8),
        critical_temp_high=15,
        critical_temp_low=0,
        exposure_time_limit=4,
        current_temperature=5,
        value_per_unit=20,
        priority_level=3,
        manufacturing_date=datetime.now() - timedelta(days=60),
        expiry_date=datetime.now() + timedelta(days=300),
        destination="Hospital_B"
    )
    
    # Insuline
    medicine3 = Medicine(
        id="MED_003",
        name="Insuline",
        medicine_type="insulin",
        batch_number="BATCH_2024_003",
        quantity=100,
        optimal_temp_range=(2, 8),
        critical_temp_high=25,
        critical_temp_low=-1,
        exposure_time_limit=3,
        current_temperature=4,
        value_per_unit=30,
        priority_level=4,
        manufacturing_date=datetime.now() - timedelta(days=90),
        expiry_date=datetime.now() + timedelta(days=90),
        destination="Hospital_C"
    )
    
    inventory.add_medicine(medicine1)
    inventory.add_medicine(medicine2)
    inventory.add_medicine(medicine3)
    
    return inventory


def example_1_basic_risk_assessment():
    """Exemple 1: Évaluation de risque de base"""
    logger.info("=" * 80)
    logger.info("EXEMPLE 1: Évaluation de risque de base")
    logger.info("=" * 80)
    
    # Créer les données
    fleet = create_sample_fleet()
    inventory = create_sample_inventory()
    
    # Initialiser le modèle de risque
    risk_model = BayesianRiskModel()
    
    # Évaluer chaque véhicule
    logger.info("\n--- Évaluation des véhicules ---")
    for vehicle in fleet.vehicles:
        assessment = risk_model.assess_vehicle_risk(vehicle)
        logger.info(f"\nVéhicule: {vehicle.id}")
        logger.info(f"  Catégorie de risque: {assessment.risk_category}")
        logger.info(f"  Score de risque: {assessment.risk_score:.3f}")
        logger.info(f"  Confiance: {assessment.confidence:.3f}")
        logger.info(f"  Probabilité de défaillance: {assessment.probability_of_failure:.3f}")
        logger.info(f"  Perte attendue: {assessment.expected_loss:.2f} €")
        logger.info(f"  Facteurs contributifs:")
        for factor, value in assessment.contributing_factors.items():
            logger.info(f"    - {factor}: {value:.3f}")
        if assessment.recommendations:
            logger.info(f"  Recommandations:")
            for rec in assessment.recommendations:
                logger.info(f"    {rec}")
    
    # Évaluer chaque médicament
    logger.info("\n--- Évaluation des médicaments ---")
    for medicine in inventory.medicines:
        assessment = risk_model.assess_medicine_risk(medicine, transit_time=4.0)
        logger.info(f"\nMédicament: {medicine.name}")
        logger.info(f"  Catégorie de risque: {assessment.risk_category}")
        logger.info(f"  Score de risque: {assessment.risk_score:.3f}")
        logger.info(f"  Probabilité de compromission: {assessment.probability_of_failure:.3f}")
        logger.info(f"  Perte attendue: {assessment.expected_loss:.2f} €")
        if assessment.recommendations:
            logger.info(f"  Recommandations:")
            for rec in assessment.recommendations:
                logger.info(f"    {rec}")


def example_2_decision_making():
    """Exemple 2: Prise de décision pour un transport"""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLE 2: Prise de décision pour un transport")
    logger.info("=" * 80)
    
    fleet = create_sample_fleet()
    inventory = create_sample_inventory()
    
    # Initialiser le moteur de décision
    decision_engine = DecisionEngine(fleet, inventory)
    
    # Conditions environnementales normales
    normal_conditions = EnvironmentConditions(
        timestamp=datetime.now(),
        location=(48.8566, 2.3522),
        ambient_temperature=15,
        humidity=60,
        wind_speed=20,
        weather_condition="normal",
        road_condition="good",
        power_availability=0.95,
        communication_quality=0.90,
        security_level="safe",
        conflict_intensity=0.0
    )
    
    # Évaluer la faisabilité de chaque combinaison véhicule-médicament
    logger.info("\n--- Évaluation des transports ---")
    for vehicle in fleet.vehicles:
        for medicine in inventory.medicines:
            decision = decision_engine.evaluate_transport_feasibility(
                vehicle,
                medicine,
                normal_conditions,
                transit_time=4.0
            )
            
            logger.info(f"\n{vehicle.id} -> {medicine.name}")
            logger.info(f"  Décision: {decision.decision_type.value}")
            logger.info(f"  Confiance: {decision.confidence:.3f}")
            logger.info(f"  Priorité: {decision.priority}/5")
            logger.info(f"  Raisonnement: {decision.reasoning}")
            logger.info(f"  Coût estimé: {decision.estimated_cost:.2f} €")
            logger.info(f"  Bénéfice estimé: {decision.estimated_benefit:.2f} €")
            if decision.alternative_actions:
                logger.info(f"  Actions alternatives: {[a.value for a in decision.alternative_actions]}")


def example_3_vehicle_optimization():
    """Exemple 3: Optimisation de l'affectation des véhicules"""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLE 3: Optimisation de l'affectation des véhicules")
    logger.info("=" * 80)
    
    fleet = create_sample_fleet()
    inventory = create_sample_inventory()
    
    decision_engine = DecisionEngine(fleet, inventory)
    
    # Optimiser les affectations
    transport_plans = decision_engine.optimize_vehicle_assignment(
        inventory.medicines
    )
    
    logger.info(f"\n{len(transport_plans)} plans de transport générés:")
    for i, plan in enumerate(transport_plans, 1):
        logger.info(f"\n--- Plan {i} ---")
        logger.info(f"Véhicule: {plan.vehicle.id}")
        logger.info(f"Médicaments: {[m.name for m in plan.medicines]}")
        logger.info(f"Score de risque: {plan.risk_score:.3f}")
        logger.info(f"Durée estimée: {plan.estimated_duration:.1f}h")
        logger.info(f"Plans de contingence:")
        for contingency in plan.contingency_plans:
            logger.info(f"  - {contingency}")


def example_4_crisis_scenario():
    """Exemple 4: Scénario de crise (zone de guerre)"""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLE 4: Scénario de crise - Zone de guerre")
    logger.info("=" * 80)
    
    fleet = create_sample_fleet()
    inventory = create_sample_inventory()
    
    decision_engine = DecisionEngine(fleet, inventory)
    
    # Conditions de zone de guerre
    war_conditions = EnvironmentConditions(
        timestamp=datetime.now(),
        location=(33.3152, 44.3661),  # Baghdad
        ambient_temperature=38,
        humidity=20,
        wind_speed=30,
        weather_condition="extreme_heat",
        road_condition="degraded",
        power_availability=0.3,
        communication_quality=0.4,
        security_level="war_zone",
        conflict_intensity=0.8
    )
    
    logger.info("\n--- Conditions environnementales ---")
    logger.info(f"Localisation: {war_conditions.location}")
    logger.info(f"Température: {war_conditions.ambient_temperature}°C")
    logger.info(f"Météo: {war_conditions.weather_condition}")
    logger.info(f"État des routes: {war_conditions.road_condition}")
    logger.info(f"Niveau de sécurité: {war_conditions.security_level}")
    logger.info(f"Intensité du conflit: {war_conditions.conflict_intensity}")
    logger.info(f"Zone extrême: {war_conditions.is_extreme_weather()}")
    logger.info(f"Zone à haut risque: {war_conditions.is_high_risk_zone()}")
    logger.info(f"Multiplicateur de risque: {war_conditions.get_risk_multiplier():.2f}x")
    
    logger.info("\n--- Décisions dans ce contexte ---")
    for medicine in inventory.medicines[:2]:  # Seulement 2 pour l'exemple
        vehicle = fleet.vehicles[0]
        decision = decision_engine.evaluate_transport_feasibility(
            vehicle,
            medicine,
            war_conditions,
            transit_time=6.0
        )
        
        logger.info(f"\nMédicament: {medicine.name}")
        logger.info(f"  Décision: {decision.decision_type.value}")
        logger.info(f"  Confiance: {decision.confidence:.3f}")
        logger.info(f"  Priorité: {decision.priority}/5")
        logger.info(f"  Raisonnement: {decision.reasoning}")
        
        # Générer un plan de contingence
        potential_events = [
            "refrigeration_failure",
            "road_blockage",
            "extreme_weather",
            "temperature_excursion"
        ]
        
        plan = decision_engine.optimize_vehicle_assignment([medicine], war_conditions)
        if plan:
            contingencies = decision_engine.generate_contingency_plan(
                plan[0],
                potential_events
            )
            
            logger.info(f"  Plans de contingence:")
            for event_type, actions in contingencies.items():
                logger.info(f"    {event_type}:")
                for action in actions[:2]:  # 2 premières actions
                    logger.info(f"      - {action}")


def example_5_simulation():
    """Exemple 5: Simulation complète"""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLE 5: Simulation complète d'une chaîne du froid")
    logger.info("=" * 80)
    
    fleet = create_sample_fleet()
    inventory = create_sample_inventory()
    
    # Mettre les véhicules en transit
    for vehicle in fleet.vehicles[:2]:
        vehicle.current_status = VehicleStatus.IN_TRANSIT
    
    logger.info("\n--- Configuration de la simulation ---")
    logger.info(f"Durée: 8 heures")
    logger.info(f"Véhicules: {len(fleet.vehicles)}")
    logger.info(f"Médicaments: {len(inventory.medicines)}")
    
    # Lancer la simulation
    logger.info("\nLancement de la simulation...")
    
    simulation = ColdChainSimulation(
        fleet=fleet,
        inventory=inventory,
        duration=8.0,
        time_step=0.5,
        random_seed=42
    )
    
    results = simulation.run()
    
    logger.info("\n--- Résultats de la simulation ---")
    logger.info(f"Entrées de log: {results['total_log_entries']}")
    logger.info(f"\nStatistiques de la flotte:")
    for key, value in results['fleet_statistics'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nStatistiques de l'inventaire:")
    for key, value in results['inventory_statistics'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nStatistiques des événements:")
    for key, value in results['event_statistics'].items():
        logger.info(f"  {key}: {value}")


def main():
    """Fonction principale exécutant tous les exemples"""
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 10 + "SYSTÈME DE DÉCISION PROBABILISTE" + " " * 36 + "║")
    logger.info("║" + " " * 10 + "Gestion de la Chaîne du Froid" + " " * 39 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    try:
        # Exécuter les exemples
        example_1_basic_risk_assessment()
        example_2_decision_making()
        example_3_vehicle_optimization()
        example_4_crisis_scenario()
        example_5_simulation()
        
        logger.info("\n" + "=" * 80)
        logger.info("Tous les exemples ont été exécutés avec succès!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
