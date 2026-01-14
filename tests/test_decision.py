"""
Tests unitaires pour le moteur de décision
"""
import pytest
from datetime import datetime, timedelta

from src.models import Vehicle, VehicleFleet, Medicine, MedicineInventory
from src.models.events import EnvironmentConditions
from src.decision import DecisionEngine, DecisionType
from src.probabilistic import BayesianRiskModel


class TestDecisionEngine:
    """Tests pour la classe DecisionEngine"""
    
    @pytest.fixture
    def setup_test_data(self):
        """Fixture pour créer des données de test"""
        # Créer un véhicule
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            age=2.0,
            last_maintenance=datetime.now() - timedelta(days=30)
        )
        
        fleet = VehicleFleet()
        fleet.add_vehicle(vehicle)
        
        # Créer un médicament
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=200,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            value_per_unit=25,
            priority_level=3
        )
        
        inventory = MedicineInventory()
        inventory.add_medicine(medicine)
        
        # Conditions normales
        conditions = EnvironmentConditions(
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
        
        return fleet, inventory, vehicle, medicine, conditions
    
    def test_decision_engine_creation(self, setup_test_data):
        """Test de création du moteur de décision"""
        fleet, inventory, _, _, _ = setup_test_data
        engine = DecisionEngine(fleet, inventory)
        
        assert engine.fleet == fleet
        assert engine.inventory == inventory
        assert isinstance(engine.risk_model, BayesianRiskModel)
    
    def test_evaluate_transport_feasibility_normal(self, setup_test_data):
        """Test d'évaluation de faisabilité en conditions normales"""
        fleet, inventory, vehicle, medicine, conditions = setup_test_data
        engine = DecisionEngine(fleet, inventory)
        
        decision = engine.evaluate_transport_feasibility(
            vehicle,
            medicine,
            conditions,
            transit_time=3.0
        )
        
        assert decision is not None
        assert decision.decision_type in [DecisionType.PROCEED, DecisionType.DELAY]
        assert 0 <= decision.confidence <= 1
        assert decision.priority >= 1
    
    def test_temperature_incompatibility(self, setup_test_data):
        """Test de détection d'incompatibilité de température"""
        fleet, inventory, vehicle, medicine, conditions = setup_test_data
        
        # Modifier le médicament pour qu'il soit incompatible
        medicine.optimal_temp_range = (-70, -60)
        
        engine = DecisionEngine(fleet, inventory)
        decision = engine.evaluate_transport_feasibility(
            vehicle,
            medicine,
            conditions
        )
        
        assert decision.decision_type == DecisionType.CHANGE_VEHICLE
    
    def test_insufficient_time(self, setup_test_data):
        """Test de détection de temps insuffisant"""
        fleet, inventory, vehicle, medicine, conditions = setup_test_data
        
        # Réduire le temps disponible
        medicine.total_exposure_time = 3.5  # Sur 4h limite
        
        engine = DecisionEngine(fleet, inventory)
        decision = engine.evaluate_transport_feasibility(
            vehicle,
            medicine,
            conditions,
            transit_time=1.0  # Dépasserait la limite
        )
        
        assert decision.decision_type == DecisionType.EMERGENCY_ACTION
    
    def test_optimize_vehicle_assignment(self, setup_test_data):
        """Test d'optimisation d'affectation"""
        fleet, inventory, _, _, conditions = setup_test_data
        engine = DecisionEngine(fleet, inventory)
        
        plans = engine.optimize_vehicle_assignment(
            inventory.medicines,
            conditions
        )
        
        assert isinstance(plans, list)
        if plans:
            assert all(hasattr(plan, 'vehicle') for plan in plans)
            assert all(hasattr(plan, 'medicines') for plan in plans)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
