"""
Connecteur pour l'intégration avec AnyLogic
"""
import pandas as pd
import json
import yaml
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ..models import Vehicle, VehicleFleet, Medicine, MedicineInventory
from ..models.events import EnvironmentConditions
from datetime import datetime

logger = logging.getLogger(__name__)


class AnyLogicConnector:
    """
    Connecteur pour importer/exporter des données avec AnyLogic
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le connecteur
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration"""
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_vehicles_from_csv(
        self,
        file_path: str,
        encoding: str = 'utf-8'
    ) -> VehicleFleet:
        """
        Charge une flotte de véhicules depuis un fichier CSV
        
        Format attendu du CSV:
        id,type,capacity,temp_min,temp_max,autonomy,speed,power,location_lat,location_lon,age
        
        Args:
            file_path: Chemin vers le fichier CSV
            encoding: Encodage du fichier
            
        Returns:
            VehicleFleet avec les véhicules chargés
        """
        logger.info(f"Chargement des véhicules depuis {file_path}")
        
        df = pd.read_csv(file_path, encoding=encoding)
        fleet = VehicleFleet()
        
        for _, row in df.iterrows():
            vehicle = Vehicle(
                id=str(row['id']),
                vehicle_type=row.get('type', 'refrigerated_truck'),
                capacity=float(row['capacity']),
                temperature_range=(float(row['temp_min']), float(row['temp_max'])),
                autonomy=float(row['autonomy']),
                avg_speed=float(row['speed']),
                refrigeration_power=float(row.get('power', 5000)),
                current_location=(
                    float(row.get('location_lat', 0)),
                    float(row.get('location_lon', 0))
                ),
                age=float(row.get('age', 0))
            )
            fleet.add_vehicle(vehicle)
        
        logger.info(f"{len(fleet.vehicles)} véhicules chargés")
        return fleet
    
    def load_medicines_from_csv(
        self,
        file_path: str,
        encoding: str = 'utf-8'
    ) -> MedicineInventory:
        """
        Charge un inventaire de médicaments depuis un fichier CSV
        
        Format attendu du CSV:
        id,name,type,batch,quantity,temp_min,temp_max,temp_crit_high,temp_crit_low,exposure_limit,value,priority
        
        Args:
            file_path: Chemin vers le fichier CSV
            encoding: Encodage du fichier
            
        Returns:
            MedicineInventory avec les médicaments chargés
        """
        logger.info(f"Chargement des médicaments depuis {file_path}")
        
        df = pd.read_csv(file_path, encoding=encoding)
        inventory = MedicineInventory()
        
        for _, row in df.iterrows():
            medicine = Medicine(
                id=str(row['id']),
                name=row['name'],
                medicine_type=row.get('type', 'vaccine'),
                batch_number=row.get('batch', 'UNKNOWN'),
                quantity=float(row['quantity']),
                optimal_temp_range=(float(row['temp_min']), float(row['temp_max'])),
                critical_temp_high=float(row['temp_crit_high']),
                critical_temp_low=float(row['temp_crit_low']),
                exposure_time_limit=float(row['exposure_limit']),
                value_per_unit=float(row.get('value', 0)),
                priority_level=int(row.get('priority', 1))
            )
            inventory.add_medicine(medicine)
        
        logger.info(f"{len(inventory.medicines)} médicaments chargés")
        return inventory
    
    def load_simulation_data_from_json(
        self,
        file_path: str
    ) -> Dict:
        """
        Charge des données de simulation depuis un fichier JSON
        
        Args:
            file_path: Chemin vers le fichier JSON
            
        Returns:
            Dictionnaire contenant les données
        """
        logger.info(f"Chargement des données de simulation depuis {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def export_results_to_csv(
        self,
        results: Dict,
        output_dir: str,
        prefix: str = "decision_"
    ):
        """
        Exporte les résultats vers des fichiers CSV
        
        Args:
            results: Résultats de simulation/décision
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Exporter le log de simulation
        if 'simulation_log' in results:
            log_df = pd.DataFrame(results['simulation_log'])
            log_file = output_path / f"{prefix}log_{timestamp}.csv"
            log_df.to_csv(log_file, index=False, encoding='utf-8')
            logger.info(f"Log exporté vers {log_file}")
        
        # Exporter les statistiques de flotte
        if 'fleet_statistics' in results:
            fleet_df = pd.DataFrame([results['fleet_statistics']])
            fleet_file = output_path / f"{prefix}fleet_stats_{timestamp}.csv"
            fleet_df.to_csv(fleet_file, index=False, encoding='utf-8')
            logger.info(f"Statistiques de flotte exportées vers {fleet_file}")
        
        # Exporter les statistiques d'inventaire
        if 'inventory_statistics' in results:
            inv_df = pd.DataFrame([results['inventory_statistics']])
            inv_file = output_path / f"{prefix}inventory_stats_{timestamp}.csv"
            inv_df.to_csv(inv_file, index=False, encoding='utf-8')
            logger.info(f"Statistiques d'inventaire exportées vers {inv_file}")
        
        # Exporter les statistiques d'événements
        if 'event_statistics' in results:
            event_df = pd.DataFrame([results['event_statistics']])
            event_file = output_path / f"{prefix}events_stats_{timestamp}.csv"
            event_df.to_csv(event_file, index=False, encoding='utf-8')
            logger.info(f"Statistiques d'événements exportées vers {event_file}")
    
    def export_decisions_to_json(
        self,
        decisions: List,
        output_file: str
    ):
        """
        Exporte les décisions vers un fichier JSON
        
        Args:
            decisions: Liste de décisions
            output_file: Fichier de sortie
        """
        decisions_data = []
        
        for decision in decisions:
            decisions_data.append({
                'decision_type': decision.decision_type.value,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'risk_score': decision.risk_assessment.risk_score,
                'risk_category': decision.risk_assessment.risk_category,
                'estimated_cost': decision.estimated_cost,
                'estimated_benefit': decision.estimated_benefit,
                'priority': decision.priority,
                'recommendations': decision.risk_assessment.recommendations
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(decisions_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Décisions exportées vers {output_file}")
    
    def create_anylogic_input_template(
        self,
        output_dir: str
    ):
        """
        Crée des templates CSV pour l'import depuis AnyLogic
        
        Args:
            output_dir: Répertoire de sortie
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Template véhicules
        vehicles_template = pd.DataFrame({
            'id': ['TRUCK_001', 'TRUCK_002'],
            'type': ['refrigerated_truck', 'cold_box'],
            'capacity': [1000, 500],
            'temp_min': [2, -80],
            'temp_max': [8, -60],
            'autonomy': [500, 400],
            'speed': [60, 55],
            'power': [5000, 8000],
            'location_lat': [48.8566, 48.8566],
            'location_lon': [2.3522, 2.3522],
            'age': [3.0, 1.5]
        })
        
        vehicles_file = output_path / 'vehicles_template.csv'
        vehicles_template.to_csv(vehicles_file, index=False, encoding='utf-8')
        logger.info(f"Template véhicules créé: {vehicles_file}")
        
        # Template médicaments
        medicines_template = pd.DataFrame({
            'id': ['MED_001', 'MED_002'],
            'name': ['Vaccin COVID-19', 'Vaccin Standard'],
            'type': ['vaccine_covid', 'vaccine_standard'],
            'batch': ['BATCH_2024_001', 'BATCH_2024_002'],
            'quantity': [200, 500],
            'temp_min': [-70, 2],
            'temp_max': [-60, 8],
            'temp_crit_high': [-50, 15],
            'temp_crit_low': [-80, 0],
            'exposure_limit': [2, 4],
            'value': [50, 20],
            'priority': [5, 3]
        })
        
        medicines_file = output_path / 'medicines_template.csv'
        medicines_template.to_csv(medicines_file, index=False, encoding='utf-8')
        logger.info(f"Template médicaments créé: {medicines_file}")


class DataConverter:
    """
    Classe utilitaire pour convertir les données entre différents formats
    """
    
    @staticmethod
    def fleet_to_dataframe(fleet: VehicleFleet) -> pd.DataFrame:
        """Convertit une flotte en DataFrame"""
        data = []
        for vehicle in fleet.vehicles:
            data.append(vehicle.to_dict())
        return pd.DataFrame(data)
    
    @staticmethod
    def inventory_to_dataframe(inventory: MedicineInventory) -> pd.DataFrame:
        """Convertit un inventaire en DataFrame"""
        data = []
        for medicine in inventory.medicines:
            data.append(medicine.to_dict())
        return pd.DataFrame(data)
    
    @staticmethod
    def events_to_dataframe(events: List) -> pd.DataFrame:
        """Convertit une liste d'événements en DataFrame"""
        data = []
        for event in events:
            data.append(event.to_dict())
        return pd.DataFrame(data)
