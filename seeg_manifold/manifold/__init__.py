"""
Manifold Learning Module

Tools for dimensionality reduction and manifold learning:
- Multiple reduction methods (PCA, UMAP, t-SNE, Isomap, PHATE)
- Intrinsic dimensionality estimation
- Multi-method comparison
"""

from .dimensionality import (
    estimate_dimensionality,
    pca_explained_variance,
    intrinsic_dim_mle,
    intrinsic_dim_correlation,
)

from .reduction import (
    reduce_dimensions,
    fit_pca,
    fit_umap,
    fit_tsne,
    fit_isomap,
    fit_phate,
)

from .comparison import (
    compare_reductions,
    compute_preservation_metrics,
    ReductionResult,
)

__all__ = [
    'estimate_dimensionality',
    'pca_explained_variance',
    'intrinsic_dim_mle',
    'intrinsic_dim_correlation',
    'reduce_dimensions',
    'fit_pca',
    'fit_umap',
    'fit_tsne',
    'fit_isomap',
    'fit_phate',
    'compare_reductions',
    'compute_preservation_metrics',
    'ReductionResult',
]
