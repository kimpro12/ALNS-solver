"""Destroy operator implementations."""

from .random_removal import random_removal
from .shaw_removal import shaw_removal
from .route_pair_destroy import route_pair_destroy
from .worst_removal import worst_removal

__all__ = [
    "random_removal",
    "shaw_removal",
    "route_pair_destroy",
    "worst_removal",
]
