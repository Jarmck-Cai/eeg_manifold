"""
Dimensionality Estimation

Methods for estimating the intrinsic dimensionality of neural data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings


def estimate_dimensionality(
    data: np.ndarray,
    methods: Optional[List[str]] = None,
    variance_threshold: float = 0.95,
    k_neighbors: int = 10,
    verbose: bool = True
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Estimate intrinsic dimensionality using multiple methods.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    methods : list of str, optional
        Methods to use. Options: 'pca_variance', 'pca_elbow', 'mle'
        Default: all methods
    variance_threshold : float
        Variance threshold for PCA-based estimation (default: 0.95)
    k_neighbors : int
        Number of neighbors for MLE method
    verbose : bool
        Print results
        
    Returns
    -------
    results : dict
        Dictionary with dimensionality estimates from each method
    """
    if methods is None:
        methods = ['pca_variance', 'pca_elbow', 'mle']
    
    results = {}
    
    # Ensure data is 2D (n_samples, n_features)
    if data.ndim == 3:
        n_epochs, n_channels, n_times = data.shape
        data = data.reshape(n_epochs, -1)
    elif data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T
    
    if verbose:
        print(f"Estimating dimensionality for data shape: {data.shape}")
    
    if 'pca_variance' in methods:
        dim, var_ratio = pca_explained_variance(data, variance_threshold)
        results['pca_variance'] = dim
        results['pca_variance_ratio'] = var_ratio
        if verbose:
            print(f"  PCA ({variance_threshold*100:.0f}% variance): {dim} dimensions")
    
    if 'pca_elbow' in methods:
        dim, eigenvalues = pca_elbow(data)
        results['pca_elbow'] = dim
        results['eigenvalues'] = eigenvalues
        if verbose:
            print(f"  PCA (elbow): {dim} dimensions")
    
    if 'mle' in methods:
        dim = intrinsic_dim_mle(data, k=k_neighbors)
        results['mle'] = dim
        if verbose:
            print(f"  MLE: {dim:.1f} dimensions")
    
    # Compute consensus estimate.
    # NumPy scalars must be handled explicitly: ``np.float64`` subclasses
    # Python ``float`` but ``np.int64`` does not subclass ``int``, so an
    # ``isinstance(v, (int, float))`` filter would silently drop the
    # integer-valued PCA estimates and leave the median equal to the MLE.
    dim_estimates = [float(v) for k, v in results.items()
                     if k in ['pca_variance', 'pca_elbow', 'mle']
                     and np.isscalar(v) and not isinstance(v, (str, bool))]
    
    if dim_estimates:
        results['consensus'] = int(np.median(dim_estimates))
        if verbose:
            print(f"  Consensus (median): {results['consensus']} dimensions")
    
    return results


def pca_explained_variance(data: np.ndarray, threshold: float = 0.95) -> Tuple[int, np.ndarray]:
    """Estimate dimensionality based on PCA explained variance."""
    n_components = min(data.shape)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dim = np.argmax(cumvar >= threshold) + 1
    
    return n_dim, pca.explained_variance_ratio_


def pca_elbow(data: np.ndarray) -> Tuple[int, np.ndarray]:
    """Estimate dimensionality using PCA elbow method."""
    n_components = min(data.shape[0], data.shape[1], 50)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    eigenvalues = pca.explained_variance_
    
    if len(eigenvalues) < 3:
        return len(eigenvalues), eigenvalues
    
    # Find elbow using curvature
    x = np.arange(len(eigenvalues))
    y = eigenvalues / eigenvalues[0]
    
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    curvature = np.abs(ddy) / (1 + dy**2)**1.5
    
    elbow = np.argmax(curvature[1:-1]) + 1
    
    return elbow + 1, eigenvalues


def intrinsic_dim_mle(data: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality using Maximum Likelihood Estimation.
    
    Based on Levina & Bickel (2004).
    """
    n_samples = data.shape[0]
    
    if k >= n_samples:
        k = n_samples - 1
    
    # Find k+1 nearest neighbors (including self)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Remove self-distance (first column)
    distances = distances[:, 1:]
    
    # Avoid log(0)
    distances = np.maximum(distances, 1e-10)
    
    # MLE estimate for each point
    # d_hat = 1 / (1/k * sum(log(r_k / r_j)))
    log_ratios = np.log(distances[:, -1:] / distances[:, :-1])
    dim_estimates = (k - 1) / np.sum(log_ratios, axis=1)
    
    # Return mean estimate
    return np.mean(dim_estimates)


def intrinsic_dim_correlation(data: np.ndarray, n_points: int = 1000) -> float:
    """
    Estimate intrinsic dimensionality using correlation dimension.
    
    Uses the Grassberger-Procaccia algorithm.
    """
    n_samples = data.shape[0]
    
    # Subsample if too many points
    if n_samples > n_points:
        idx = np.random.choice(n_samples, n_points, replace=False)
        data = data[idx]
        n_samples = n_points
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(data)
    
    # Remove zeros
    distances = distances[distances > 0]
    
    if len(distances) == 0:
        return 0.0
    
    # Estimate correlation dimension using linear regression on log-log plot
    r_values = np.logspace(np.log10(np.percentile(distances, 1)),
                          np.log10(np.percentile(distances, 50)), 20)
    
    C_r = np.array([np.mean(distances < r) for r in r_values])
    
    # Filter valid points
    valid = (C_r > 0) & (C_r < 1)
    if np.sum(valid) < 2:
        return 0.0
    
    log_r = np.log(r_values[valid])
    log_C = np.log(C_r[valid])
    
    # Linear regression
    slope, _ = np.polyfit(log_r, log_C, 1)
    
    return slope
