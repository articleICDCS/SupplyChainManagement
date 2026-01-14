"""
Script pour générer des templates et données d'exemple
"""
from src.integration import AnyLogicConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Génère les templates et données d'exemple"""
    logger.info("Génération des templates de données...")
    
    connector = AnyLogicConnector()
    
    # Créer les templates CSV
    connector.create_anylogic_input_template("data/templates")
    
    logger.info("✓ Templates générés dans data/templates/")
    logger.info("  - vehicles_template.csv")
    logger.info("  - medicines_template.csv")
    logger.info("\nVous pouvez maintenant :")
    logger.info("1. Modifier ces templates avec vos données")
    logger.info("2. Les importer depuis AnyLogic")
    logger.info("3. Ou les utiliser directement dans Python")


if __name__ == "__main__":
    main()
