"""
Tests unitaires pour les modèles de médicaments
"""
import pytest
from datetime import datetime, timedelta

from src.models.medicine import Medicine, MedicineInventory, MedicineStatus


class TestMedicine:
    """Tests pour la classe Medicine"""
    
    def test_medicine_creation(self):
        """Test de création d'un médicament"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4
        )
        
        assert medicine.id == "MED_001"
        assert medicine.quantity == 100
        assert medicine.current_status == MedicineStatus.SAFE
    
    def test_temperature_optimal(self):
        """Test de vérification de température optimale"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            current_temperature=5
        )
        
        assert medicine.is_temperature_optimal() is True
        
        medicine.current_temperature = 10
        assert medicine.is_temperature_optimal() is False
    
    def test_temperature_critical(self):
        """Test de détection de température critique"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            current_temperature=5
        )
        
        assert medicine.is_temperature_critical() is False
        
        medicine.current_temperature = 20
        assert medicine.is_temperature_critical() is True
        
        medicine.current_temperature = -5
        assert medicine.is_temperature_critical() is True
    
    def test_temperature_exposure_recording(self):
        """Test de l'enregistrement d'exposition"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4
        )
        
        medicine.record_temperature_exposure(12, 1.0)
        
        assert len(medicine.temperature_history) == 1
        assert medicine.total_exposure_time == 1.0
        assert medicine.current_status == MedicineStatus.AT_RISK
    
    def test_remaining_safe_time(self):
        """Test du calcul du temps restant"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            total_exposure_time=2.0
        )
        
        assert medicine.get_remaining_safe_time() == 2.0
    
    def test_total_value(self):
        """Test du calcul de valeur totale"""
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            value_per_unit=25
        )
        
        assert medicine.get_total_value() == 2500


class TestMedicineInventory:
    """Tests pour la classe MedicineInventory"""
    
    def test_inventory_creation(self):
        """Test de création d'inventaire"""
        inventory = MedicineInventory()
        assert len(inventory.medicines) == 0
    
    def test_add_medicine(self):
        """Test d'ajout de médicament"""
        inventory = MedicineInventory()
        medicine = Medicine(
            id="MED_001",
            name="Test Vaccine",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4
        )
        
        inventory.add_medicine(medicine)
        assert len(inventory.medicines) == 1
    
    def test_get_high_priority_medicines(self):
        """Test de récupération des médicaments prioritaires"""
        inventory = MedicineInventory()
        
        medicine1 = Medicine(
            id="MED_001",
            name="High Priority",
            medicine_type="vaccine",
            batch_number="BATCH_001",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            priority_level=5
        )
        
        medicine2 = Medicine(
            id="MED_002",
            name="Low Priority",
            medicine_type="vaccine",
            batch_number="BATCH_002",
            quantity=100,
            optimal_temp_range=(2, 8),
            critical_temp_high=15,
            critical_temp_low=0,
            exposure_time_limit=4,
            priority_level=2
        )
        
        inventory.add_medicine(medicine1)
        inventory.add_medicine(medicine2)
        
        high_priority = inventory.get_high_priority_medicines(min_priority=4)
        assert len(high_priority) == 1
        assert high_priority[0].id == "MED_001"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
