"""
Tests unitaires pour les modèles de véhicules
"""
import pytest
from datetime import datetime, timedelta

from src.models.vehicle import Vehicle, VehicleFleet, VehicleStatus, RefrigerationStatus


class TestVehicle:
    """Tests pour la classe Vehicle"""
    
    def test_vehicle_creation(self):
        """Test de création d'un véhicule"""
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000
        )
        
        assert vehicle.id == "TEST_001"
        assert vehicle.capacity == 1000
        assert vehicle.current_status == VehicleStatus.IDLE
        assert vehicle.refrigeration_status == RefrigerationStatus.OPERATIONAL
    
    def test_remaining_capacity(self):
        """Test du calcul de capacité restante"""
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            current_load=400
        )
        
        assert vehicle.get_remaining_capacity() == 600
    
    def test_temperature_compliance(self):
        """Test de la vérification de conformité de température"""
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            current_temperature=5
        )
        
        assert vehicle.is_temperature_compliant() is True
        
        vehicle.current_temperature = 10
        assert vehicle.is_temperature_compliant() is False
    
    def test_risk_score_calculation(self):
        """Test du calcul de score de risque"""
        # Véhicule récent et bien entretenu
        vehicle_low_risk = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            age=1.0,
            last_maintenance=datetime.now() - timedelta(days=30),
            breakdown_count=0
        )
        
        # Véhicule ancien et mal entretenu
        vehicle_high_risk = Vehicle(
            id="TEST_002",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            age=10.0,
            last_maintenance=datetime.now() - timedelta(days=200),
            breakdown_count=5,
            refrigeration_status=RefrigerationStatus.DEGRADED
        )
        
        assert vehicle_low_risk.get_risk_score() < vehicle_high_risk.get_risk_score()
        assert vehicle_low_risk.get_risk_score() < 0.5
        assert vehicle_high_risk.get_risk_score() > 0.5


class TestVehicleFleet:
    """Tests pour la classe VehicleFleet"""
    
    def test_fleet_creation(self):
        """Test de création d'une flotte"""
        fleet = VehicleFleet()
        assert len(fleet.vehicles) == 0
    
    def test_add_vehicle(self):
        """Test d'ajout de véhicule"""
        fleet = VehicleFleet()
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000
        )
        
        fleet.add_vehicle(vehicle)
        assert len(fleet.vehicles) == 1
    
    def test_get_available_vehicles(self):
        """Test de récupération des véhicules disponibles"""
        fleet = VehicleFleet()
        
        vehicle1 = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            current_status=VehicleStatus.IDLE
        )
        
        vehicle2 = Vehicle(
            id="TEST_002",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000,
            current_status=VehicleStatus.IN_TRANSIT
        )
        
        fleet.add_vehicle(vehicle1)
        fleet.add_vehicle(vehicle2)
        
        available = fleet.get_available_vehicles()
        assert len(available) == 1
        assert available[0].id == "TEST_001"
    
    def test_fleet_statistics(self):
        """Test des statistiques de flotte"""
        fleet = VehicleFleet()
        
        vehicle = Vehicle(
            id="TEST_001",
            vehicle_type="refrigerated_truck",
            capacity=1000,
            temperature_range=(2, 8),
            autonomy=500,
            avg_speed=60,
            refrigeration_power=5000
        )
        
        fleet.add_vehicle(vehicle)
        stats = fleet.get_fleet_statistics()
        
        assert stats['total_vehicles'] == 1
        assert stats['available'] == 1
        assert 'average_risk_score' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
