"""Operator-level pieces of the Laniakea Barnes-Hut study.

Only what the paper's operator claims depend on is shipped here; the CAMELS
adapters, graph baselines, loss functions (GUDHI-based) and the training
pipeline live in the private development repository.
"""

from .grad_utils import (
    compute_acceleration,
    soften_distance_sq,
    inverse_square_acceleration,
)
from .barnes_hut import (
    BarnesHutTree,
    multipole_moments_from_tree,
    exact_force,
    iter_interaction_batches,
)
from .exact_field import exact_force_and_tidal, exact_tidal
from .gravity import GravitationalMessage
from .hmmp import HMMP, HMMPLayer

__all__ = [
    "compute_acceleration",
    "soften_distance_sq",
    "inverse_square_acceleration",
    "BarnesHutTree",
    "multipole_moments_from_tree",
    "exact_force",
    "iter_interaction_batches",
    "exact_force_and_tidal",
    "exact_tidal",
    "GravitationalMessage",
    "HMMP",
    "HMMPLayer",
]
