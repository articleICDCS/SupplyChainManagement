"""
Module principal d'initialisation
"""
from . import models
from . import simulation
from . import decision
from . import probabilistic
from . import integration
from . import utils

__version__ = "1.0.0"

__all__ = [
    'models',
    'simulation',
    'decision',
    'probabilistic',
    'integration',
    'utils'
]
