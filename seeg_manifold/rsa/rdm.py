"""
Representational Dissimilarity Matrix (RDM)

Compute and analyze representational dissimilarity matrices for RSA.
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr


def compute_rdm(
    data: np.ndarray,
    metric: str = 'correlation',
    normalize: bool = True
) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_conditions, n_features) or (n_conditions, n_channels, n_times)
    metric : str
        Distance metric: 'correlation', 'euclidean', 'cosine', 'mahalanobis'
    normalize : bool
        Whether to normalize the RDM to [0, 1] range
        
    Returns
    -------
    rdm : np.ndarray
        RDM matrix (n_conditions, n_conditions)
    """
    # Flatten if 3D
    if data.ndim == 3:
        n_conditions = data.shape[0]
        data = data.reshape(n_conditions, -1)
    
    if metric == 'correlation':
        # 1 - correlation (so 0 = identical, 2 = opposite)
        rdm = 1 - np.corrcoef(data)
    elif metric == 'cosine':
        # Normalize each row
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data_norm = data / (norms + 1e-10)
        rdm = 1 - (data_norm @ data_norm.T)
    else:
        # Use scipy pdist for other metrics
        distances = pdist(data, metric=metric)
        rdm = squareform(distances)
    
    # Ensure diagonal is 0
    np.fill_diagonal(rdm, 0)
    
    if normalize:
        rdm_max = np.max(rdm)
        if rdm_max > 0:
            rdm = rdm / rdm_max
    
    return rdm


def compute_rdm_timeseries(
    data: np.ndarray,
    metric: str = 'correlation',
    window_size: Optional[int] = None,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RDM for each timepoint or time window.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_conditions, n_channels, n_times)
    metric : str
        Distance metric
    window_size : int, optional
        Window size for temporal averaging (None = single timepoint)
    step : int
        Step size between windows
        
    Returns
    -------
    rdms : np.ndarray
        RDM timeseries (n_times, n_conditions, n_conditions)
    time_indices : np.ndarray
        Time indices for each RDM
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")
    
    n_conditions, n_channels, n_times = data.shape
    
    if window_size is None:
        window_size = 1
    
    n_windows = (n_times - window_size) // step + 1
    rdms = np.zeros((n_windows, n_conditions, n_conditions))
    time_indices = np.zeros(n_windows, dtype=int)
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        
        # Average within window
        window_data = data[:, :, start:end].mean(axis=-1)
        
        rdms[i] = compute_rdm(window_data, metric=metric, normalize=True)
        time_indices[i] = start + window_size // 2
    
    return rdms, time_indices


def compare_rdms(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    method: str = 'spearman'
) -> Tuple[float, float]:
    """
    Compare two RDMs using correlation.
    
    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM
    rdm2 : np.ndarray
        Second RDM
    method : str
        Comparison method: 'spearman', 'pearson', 'kendall'
        
    Returns
    -------
    correlation : float
        Correlation between RDMs
    p_value : float
        P-value for the correlation
    """
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(rdm1, k=1)
    
    rdm1_vec = rdm1[triu_idx]
    rdm2_vec = rdm2[triu_idx]
    
    if method == 'spearman':
        corr, pval = spearmanr(rdm1_vec, rdm2_vec)
    elif method == 'pearson':
        corr, pval = pearsonr(rdm1_vec, rdm2_vec)
    elif method == 'kendall':
        from scipy.stats import kendalltau
        corr, pval = kendalltau(rdm1_vec, rdm2_vec)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr, pval


def compare_rdm_to_models(
    data_rdm: np.ndarray,
    model_rdms: Dict[str, np.ndarray],
    method: str = 'spearman'
) -> Dict[str, Dict[str, float]]:
    """
    Compare data RDM to multiple model RDMs.
    
    Parameters
    ----------
    data_rdm : np.ndarray
        Data RDM
    model_rdms : dict
        Dictionary mapping model names to model RDMs
    method : str
        Comparison method
        
    Returns
    -------
    results : dict
        Dictionary with correlation and p-value for each model
    """
    results = {}
    
    for model_name, model_rdm in model_rdms.items():
        corr, pval = compare_rdms(data_rdm, model_rdm, method=method)
        results[model_name] = {
            'correlation': corr,
            'p_value': pval
        }
    
    return results


def create_model_rdm(
    labels: np.ndarray,
    model_type: str = 'categorical'
) -> np.ndarray:
    """
    Create a model RDM from labels.
    
    Parameters
    ----------
    labels : np.ndarray
        Condition labels
    model_type : str
        Model type: 'categorical' (same/different), 'ordinal' (distance)
        
    Returns
    -------
    model_rdm : np.ndarray
        Model RDM
    """
    n = len(labels)
    model_rdm = np.zeros((n, n))
    
    if model_type == 'categorical':
        # 0 if same category, 1 if different
        for i in range(n):
            for j in range(n):
                if labels[i] != labels[j]:
                    model_rdm[i, j] = 1
    
    elif model_type == 'ordinal':
        # Absolute difference in labels (assumes numeric labels)
        labels = np.asarray(labels, dtype=float)
        for i in range(n):
            for j in range(n):
                model_rdm[i, j] = np.abs(labels[i] - labels[j])
        
        # Normalize
        max_val = np.max(model_rdm)
        if max_val > 0:
            model_rdm = model_rdm / max_val
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_rdm


def rdm_mds(
    rdm: np.ndarray,
    n_components: int = 2
) -> np.ndarray:
    """
    Apply MDS to RDM for visualization.
    
    Parameters
    ----------
    rdm : np.ndarray
        RDM matrix
    n_components : int
        Number of MDS dimensions
        
    Returns
    -------
    embedding : np.ndarray
        MDS embedding (n_conditions, n_components)
    """
    import inspect

    from sklearn.manifold import MDS

    # scikit-learn 1.9 renamed MDS's ``dissimilarity`` to ``metric`` (and
    # the old boolean ``metric`` to ``metric_mds``); ``dissimilarity`` is
    # removed in 1.10. Pick the spelling this version accepts so the
    # function works across the supported range.
    params = inspect.signature(MDS.__init__).parameters
    kwargs = {}
    if 'metric_mds' in params:
        kwargs['metric'] = 'precomputed'
    else:
        kwargs['dissimilarity'] = 'precomputed'
    if 'init' in params:
        # The default is changing; pin it so results stay comparable.
        kwargs['init'] = 'random'

    mds = MDS(
        n_components=n_components,
        random_state=42,
        normalized_stress='auto',
        **kwargs
    )

    embedding = mds.fit_transform(rdm)
    return embedding

