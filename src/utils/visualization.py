"""
Utilitaires pour l'analyse et la visualisation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def setup_plotting_style():
    """Configure le style des graphiques"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_risk_distribution(
    risk_assessments: List,
    title: str = "Distribution des risques",
    save_path: Optional[str] = None
):
    """
    Trace la distribution des scores de risque
    
    Args:
        risk_assessments: Liste des évaluations de risque
        title: Titre du graphique
        save_path: Chemin pour sauvegarder le graphique
    """
    setup_plotting_style()
    
    risk_scores = [ra.risk_score for ra in risk_assessments]
    categories = [ra.risk_category for ra in risk_assessments]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme des scores
    ax1.hist(risk_scores, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Score de risque')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des scores de risque')
    ax1.axvline(np.mean(risk_scores), color='red', linestyle='--', label=f'Moyenne: {np.mean(risk_scores):.3f}')
    ax1.legend()
    
    # Diagramme à barres des catégories
    category_counts = pd.Series(categories).value_counts()
    ax2.bar(category_counts.index, category_counts.values)
    ax2.set_xlabel('Catégorie de risque')
    ax2.set_ylabel('Nombre')
    ax2.set_title('Répartition par catégorie')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {save_path}")
    
    plt.show()


def plot_simulation_timeline(
    simulation_log: pd.DataFrame,
    metric: str = 'vehicle_temperature',
    save_path: Optional[str] = None
):
    """
    Trace l'évolution temporelle d'une métrique de simulation
    
    Args:
        simulation_log: DataFrame du log de simulation
        metric: Métrique à tracer
        save_path: Chemin pour sauvegarder
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for vehicle_id in simulation_log['vehicle_id'].unique():
        vehicle_data = simulation_log[simulation_log['vehicle_id'] == vehicle_id]
        ax.plot(
            vehicle_data['simulation_time'],
            vehicle_data[metric],
            label=vehicle_id,
            marker='o',
            markersize=3
        )
    
    ax.set_xlabel('Temps de simulation (heures)')
    ax.set_ylabel(metric)
    ax.set_title(f'Évolution de {metric} au cours du temps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {save_path}")
    
    plt.show()


def plot_risk_factors_comparison(
    risk_assessment: Dict,
    title: str = "Facteurs de risque",
    save_path: Optional[str] = None
):
    """
    Trace un diagramme en barres des facteurs de risque
    
    Args:
        risk_assessment: Évaluation de risque avec facteurs contributifs
        title: Titre du graphique
        save_path: Chemin pour sauvegarder
    """
    setup_plotting_style()
    
    factors = risk_assessment.contributing_factors
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if v > 0.7 else 'orange' if v > 0.5 else 'green' for v in factors.values()]
    
    bars = ax.barh(list(factors.keys()), list(factors.values()), color=colors, alpha=0.7)
    ax.set_xlabel('Score de risque')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars, factors.values())):
        ax.text(value + 0.02, i, f'{value:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {save_path}")
    
    plt.show()


def generate_summary_report(
    results: Dict,
    output_file: str
):
    """
    Génère un rapport de synthèse en Markdown
    
    Args:
        results: Résultats de simulation/décision
        output_file: Fichier de sortie
    """
    from datetime import datetime
    
    report = []
    report.append("# Rapport de Simulation - Chaîne du Froid\n")
    report.append(f"*Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append("\n---\n")
    
    # Statistiques générales
    report.append("\n## 📊 Statistiques Générales\n")
    if 'duration' in results:
        report.append(f"- **Durée de simulation**: {results['duration']} heures\n")
    if 'total_log_entries' in results:
        report.append(f"- **Entrées de log**: {results['total_log_entries']}\n")
    
    # Statistiques de flotte
    if 'fleet_statistics' in results:
        report.append("\n## 🚚 Flotte de Véhicules\n")
        stats = results['fleet_statistics']
        for key, value in stats.items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
    
    # Statistiques d'inventaire
    if 'inventory_statistics' in results:
        report.append("\n## 💊 Inventaire de Médicaments\n")
        stats = results['inventory_statistics']
        for key, value in stats.items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
    
    # Statistiques d'événements
    if 'event_statistics' in results:
        report.append("\n## ⚠️ Événements Critiques\n")
        stats = results['event_statistics']
        for key, value in stats.items():
            if isinstance(value, dict):
                report.append(f"\n### {key.replace('_', ' ').title()}\n")
                for k, v in value.items():
                    report.append(f"  - {k}: {v}\n")
            else:
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
    
    # Écrire le rapport
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    logger.info(f"Rapport généré: {output_file}")


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calcule des statistiques descriptives
    
    Args:
        data: Liste de valeurs numériques
        
    Returns:
        Dictionnaire de statistiques
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75))
    }
