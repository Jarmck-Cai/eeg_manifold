"""
Dimensionality Reduction Methods

Unified interface for multiple reduction techniques.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import warnings


def reduce_dimensions(
    data: np.ndarray,
    method: str = 'pca',
    n_components: int = 3,
    samples_axis: int = 0,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality using specified method.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features) or (n_epochs, n_channels, n_times)
    method : str
        Reduction method: 'pca', 'umap', 'tsne', 'isomap', 'phate'
    n_components : int
        Target dimensionality
    samples_axis : int
        Axis containing samples. For 2D data: 0 means (n_samples, n_features),
        1 means (n_features, n_samples). Ignored for 3D data. Default: 0
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    embedding : np.ndarray
        Reduced data (n_samples, n_components)
    """
    # Validate and reshape data
    if data.ndim == 3:
        # Epoched data: (n_epochs, n_channels, n_times) -> (n_epochs, n_channels*n_times)
        n_epochs, n_channels, n_times = data.shape
        data = data.reshape(n_epochs, -1)
    elif data.ndim == 2:
        if samples_axis == 1:
            # User explicitly specified features are on axis 0
            data = data.T
        elif data.shape[0] < data.shape[1]:
            # Warn about ambiguous shape but don't auto-transpose
            warnings.warn(
                f"Data shape {data.shape} has more features ({data.shape[1]}) than samples "
                f"({data.shape[0]}). If samples are along axis 1, set samples_axis=1. "
                f"Proceeding with current orientation (n_samples={data.shape[0]})."
            )
    elif data.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    method = method.lower()
    
    if method == 'pca':
        return fit_pca(data, n_components, **kwargs)
    elif method == 'umap':
        return fit_umap(data, n_components, **kwargs)
    elif method == 'tsne':
        return fit_tsne(data, n_components, **kwargs)
    elif method == 'isomap':
        return fit_isomap(data, n_components, **kwargs)
    elif method == 'phate':
        return fit_phate(data, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Options: pca, umap, tsne, isomap, phate")


def fit_pca(
    data: np.ndarray,
    n_components: int = 3,
    whiten: bool = False,
    return_model: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Fit PCA and transform data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Number of components
    whiten : bool
        Whether to whiten the data
    return_model : bool
        If True, also return the fitted PCA model
        
    Returns
    -------
    embedding : np.ndarray
        Transformed data
    model : PCA (optional)
        Fitted model if return_model=True
    """
    pca = PCA(n_components=n_components, whiten=whiten)
    embedding = pca.fit_transform(data)
    
    if return_model:
        return embedding, pca
    return embedding


def fit_umap(
    data: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Fit UMAP and transform data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Number of components
    n_neighbors : int
        Number of neighbors for local structure
    min_dist : float
        Minimum distance between points in embedding
    metric : str
        Distance metric
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    embedding : np.ndarray
        Transformed data
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Run: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs
    )
    
    embedding = reducer.fit_transform(data)
    return embedding


def fit_tsne(
    data: np.ndarray,
    n_components: int = 3,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = 'auto',
    n_iter: int = 1000,
    random_state: int = 42,
    init: str = 'pca',
    **kwargs
) -> np.ndarray:
    """
    Fit t-SNE and transform data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Number of components (2 or 3)
    perplexity : float
        Perplexity parameter
    learning_rate : float or 'auto'
        Learning rate
    n_iter : int
        Number of iterations (max_iter in sklearn >= 1.2)
    random_state : int
        Random state
    init : str
        Initialization method ('random', 'pca')
        
    Returns
    -------
    embedding : np.ndarray
        Transformed data
    """
    # t-SNE works best with 2 or 3 components
    if n_components > 3:
        import warnings
        warnings.warn("t-SNE is typically used with 2-3 components. "
                     "Consider using UMAP for higher dimensions.")
    
    # Adjust perplexity if needed
    n_samples = data.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)
    
    # sklearn >= 1.2 renamed n_iter to max_iter
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state,
        init=init,
        **kwargs
    )
    
    embedding = tsne.fit_transform(data)
    return embedding


def fit_isomap(
    data: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 10,
    **kwargs
) -> np.ndarray:
    """
    Fit Isomap and transform data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Number of components
    n_neighbors : int
        Number of neighbors for graph construction
        
    Returns
    -------
    embedding : np.ndarray
        Transformed data
    """
    isomap = Isomap(
        n_components=n_components,
        n_neighbors=n_neighbors,
        **kwargs
    )
    
    embedding = isomap.fit_transform(data)
    return embedding


def fit_phate(
    data: np.ndarray,
    n_components: int = 3,
    knn: int = 5,
    decay: int = 40,
    t: Union[int, str] = 'auto',
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Fit PHATE and transform data.
    
    PHATE is particularly good for visualizing trajectories and
    branching structures in data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_features)
    n_components : int
        Number of components
    knn : int
        Number of nearest neighbors
    decay : int
        Decay parameter for kernel
    t : int or 'auto'
        Diffusion time
    random_state : int
        Random state
        
    Returns
    -------
    embedding : np.ndarray
        Transformed data
    """
    try:
        import phate
    except ImportError:
        raise ImportError("PHATE not installed. Run: pip install phate")
    
    reducer = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        t=t,
        random_state=random_state,
        **kwargs
    )
    
    embedding = reducer.fit_transform(data)
    return embedding
