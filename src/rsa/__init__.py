"""
Representational Similarity Analysis (RSA) Module

Tools for analyzing neural representations:
- Representational Dissimilarity Matrices (RDMs)
- RDM comparison and model testing
- Temporal RSA
"""

from .rdm import (
    compute_rdm,
    compute_rdm_timeseries,
    compare_rdms,
    compare_rdm_to_models,
    create_model_rdm,
    rdm_mds,
)

__all__ = [
    'compute_rdm',
    'compute_rdm_timeseries',
    'compare_rdms',
    'compare_rdm_to_models',
    'create_model_rdm',
    'rdm_mds',
]

