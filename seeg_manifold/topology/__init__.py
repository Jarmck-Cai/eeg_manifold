"""
Topological Data Analysis Module

Tools for analyzing topological structure in neural data:
- Persistent homology
- Betti curves
- Persistence landscapes and images
- Diagram distances
"""

from .persistent_homology import (
    compute_persistence_diagram,
    compute_persistence_landscape,
    compute_betti_curve,
    persistence_statistics,
    bottleneck_distance,
    wasserstein_distance,
    persistence_image,
)

__all__ = [
    'compute_persistence_diagram',
    'compute_persistence_landscape',
    'compute_betti_curve',
    'persistence_statistics',
    'bottleneck_distance',
    'wasserstein_distance',
    'persistence_image',
]

