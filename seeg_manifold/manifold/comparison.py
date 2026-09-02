"""
Multi-method Comparison for Dimensionality Reduction

Compare results across different reduction methods to assess robustness.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import warnings

from .reduction import reduce_dimensions


@dataclass
class ReductionResult:
    """Container for dimensionality reduction results."""
    method: str
    embedding: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def compare_reductions(
    data: np.ndarray,
    methods: Optional[List[str]] = None,
    n_components: int = 3,
    compute_metrics: bool = True,
    method_params: Optional[Dict[str, Dict]] = None,
    verbose: bool = True
) -> Dict[str, ReductionResult]:
    """
    Apply multiple reduction methods and compare results.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    methods : list of str, optional
        Methods to compare. Default: ['pca', 'umap', 'tsne', 'isomap']
    n_components : int
        Target dimensionality for all methods
    compute_metrics : bool
        Whether to compute preservation metrics
    method_params : dict, optional
        Method-specific parameters, e.g., {'umap': {'n_neighbors': 20}}
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary mapping method names to ReductionResult objects
        
    Examples
    --------
    >>> results = compare_reductions(data, methods=['pca', 'umap'])
    >>> pca_embedding = results['pca'].embedding
    >>> print(results['umap'].metrics)
    """
    if methods is None:
        methods = ['pca', 'umap', 'tsne', 'isomap']
    
    if method_params is None:
        method_params = {}
    
    # Reshape if needed
    if data.ndim == 3:
        n_epochs, n_channels, n_times = data.shape
        data = data.reshape(n_epochs, -1)
        if verbose:
            print(f"Reshaped data from {(n_epochs, n_channels, n_times)} to {data.shape}")
    elif data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T
        if verbose:
            print(f"Transposed data to {data.shape}")
    
    results = {}
    
    # Compute original distances for metrics
    if compute_metrics:
        if verbose:
            print("Computing original pairwise distances...")
        original_distances = pdist(data)
    
    for method in methods:
        if verbose:
            print(f"Fitting {method.upper()}...")
        
        try:
            params = method_params.get(method, {})
            embedding = reduce_dimensions(
                data, 
                method=method, 
                n_components=n_components,
                **params
            )
            
            result = ReductionResult(
                method=method,
                embedding=embedding,
                params={'n_components': n_components, **params}
            )
            
            # Compute metrics
            if compute_metrics:
                embedded_distances = pdist(embedding)
                metrics = compute_preservation_metrics(
                    original_distances, 
                    embedded_distances
                )
                result.metrics = metrics
                
                if verbose:
                    print(f"  Distance correlation: {metrics['distance_correlation']:.3f}")
            
            results[method] = result
            
        except Exception as e:
            warnings.warn(f"Failed to fit {method}: {e}")
            continue
    
    if verbose:
        print("\nComparison complete!")
        if compute_metrics:
            print("\nDistance preservation (Spearman correlation):")
            for method, result in results.items():
                corr = result.metrics.get('distance_correlation', 'N/A')
                if isinstance(corr, float):
                    print(f"  {method}: {corr:.3f}")
    
    return results


def compute_preservation_metrics(
    original_distances: np.ndarray,
    embedded_distances: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for how well the embedding preserves structure.
    
    Parameters
    ----------
    original_distances : np.ndarray
        Pairwise distances in original space (from pdist)
    embedded_distances : np.ndarray
        Pairwise distances in embedded space (from pdist)
        
    Returns
    -------
    metrics : dict
        Dictionary of preservation metrics
    """
    metrics = {}
    
    # Distance correlation (Spearman)
    corr, pval = spearmanr(original_distances, embedded_distances)
    metrics['distance_correlation'] = corr
    metrics['distance_correlation_pval'] = pval
    
    # Trustworthiness and continuity could be added here
    # These measure local neighborhood preservation
    
    return metrics


def compute_neighborhood_preservation(
    data_original: np.ndarray,
    data_embedded: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute neighborhood preservation metrics.
    
    Parameters
    ----------
    data_original : np.ndarray
        Original data (n_samples, n_features_orig)
    data_embedded : np.ndarray
        Embedded data (n_samples, n_features_embed)
    k : int
        Number of neighbors to consider
        
    Returns
    -------
    metrics : dict
        Trustworthiness and continuity scores
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples = data_original.shape[0]
    
    # Find k nearest neighbors in both spaces
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(data_original)
    nbrs_embed = NearestNeighbors(n_neighbors=k+1).fit(data_embedded)
    
    _, indices_orig = nbrs_orig.kneighbors(data_original)
    _, indices_embed = nbrs_embed.kneighbors(data_embedded)
    
    # Remove self (first neighbor)
    indices_orig = indices_orig[:, 1:]
    indices_embed = indices_embed[:, 1:]
    
    # Precompute all pairwise distances for efficiency
    from scipy.spatial.distance import cdist
    dist_orig = cdist(data_original, data_original)
    dist_embed = cdist(data_embedded, data_embedded)
    
    # Precompute ranks for all points
    # ranks_orig[i, j] = rank of point j when sorted by distance from point i
    ranks_orig = np.argsort(np.argsort(dist_orig, axis=1), axis=1)
    ranks_embed = np.argsort(np.argsort(dist_embed, axis=1), axis=1)
    
    # Trustworthiness: are embedded neighbors also neighbors in original space?
    trust_sum = 0
    for i in range(n_samples):
        orig_set = set(indices_orig[i])
        for neighbor in indices_embed[i]:
            if neighbor not in orig_set:
                # Rank of 'neighbor' from point 'i' in original space
                rank = ranks_orig[i, neighbor]
                trust_sum += max(0, rank - k)
    
    normalization = 2 / (n_samples * k * (2 * n_samples - 3 * k - 1))
    trustworthiness = 1 - normalization * trust_sum
    
    # Continuity: are original neighbors still neighbors in embedded space?
    cont_sum = 0
    for i in range(n_samples):
        embed_set = set(indices_embed[i])
        for neighbor in indices_orig[i]:
            if neighbor not in embed_set:
                # Rank of 'neighbor' from point 'i' in embedded space
                rank = ranks_embed[i, neighbor]
                cont_sum += max(0, rank - k)
    
    continuity = 1 - normalization * cont_sum
    
    return {
        'trustworthiness': trustworthiness,
        'continuity': continuity,
        'k_neighbors': k
    }
