"""
Connectivity Feature Extraction

Compute functional connectivity measures between SEEG channels.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
from scipy import signal
from scipy.stats import pearsonr


def compute_correlation_matrix(
    data: np.ndarray,
    method: str = 'pearson'
) -> np.ndarray:
    """
    Compute correlation matrix between channels.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints) or (n_epochs, n_channels, n_timepoints)
    method : str
        Correlation method: 'pearson' or 'spearman'
        
    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix (n_channels, n_channels)
    """
    if data.ndim == 3:
        # Average correlation across epochs
        n_epochs, n_channels, _ = data.shape
        corr_sum = np.zeros((n_channels, n_channels))
        
        for ep in range(n_epochs):
            if method == 'pearson':
                corr_sum += np.corrcoef(data[ep])
            elif method == 'spearman':
                from scipy.stats import spearmanr
                corr_sum += spearmanr(data[ep].T)[0]
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return corr_sum / n_epochs
    
    elif data.ndim == 2:
        if method == 'pearson':
            return np.corrcoef(data)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            return spearmanr(data.T)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def compute_coherence(
    data: np.ndarray,
    sfreq: float,
    freq_band: Optional[Tuple[float, float]] = None,
    nperseg: Optional[int] = None
) -> np.ndarray:
    """
    Compute coherence between all channel pairs.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    freq_band : tuple, optional
        Frequency band to average coherence over (low, high)
    nperseg : int, optional
        Segment length for coherence computation
        
    Returns
    -------
    coherence : np.ndarray
        Coherence matrix (n_channels, n_channels)
    """
    if data.ndim == 3:
        # Concatenate epochs for coherence
        n_epochs, n_channels, n_times = data.shape
        data = data.transpose(1, 0, 2).reshape(n_channels, -1)
    
    n_channels = data.shape[0]
    
    if nperseg is None:
        nperseg = min(256, data.shape[1] // 4)
    
    coherence_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        coherence_matrix[i, i] = 1.0
        for j in range(i + 1, n_channels):
            freqs, coh = signal.coherence(
                data[i], data[j], 
                fs=sfreq, nperseg=nperseg
            )
            
            if freq_band is not None:
                idx = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
                mean_coh = np.mean(coh[idx])
            else:
                mean_coh = np.mean(coh)
            
            coherence_matrix[i, j] = mean_coh
            coherence_matrix[j, i] = mean_coh
    
    return coherence_matrix


def compute_phase_locking_value(
    data: np.ndarray,
    sfreq: float,
    freq_band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) between channels.
    
    PLV measures phase synchronization independently of amplitude.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints) or (n_epochs, n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    freq_band : tuple
        Frequency band (low, high) for phase extraction
        
    Returns
    -------
    plv : np.ndarray
        PLV matrix (n_channels, n_channels)
    """
    from scipy.signal import hilbert
    
    # Bandpass filter to the frequency band
    from src.preprocessing.filters import bandpass_filter
    
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # Add epoch dimension
    
    n_epochs, n_channels, n_times = data.shape
    
    # Filter and get analytic signal
    filtered = bandpass_filter(data, sfreq, freq_band[0], freq_band[1], axis=-1)
    analytic = hilbert(filtered, axis=-1)
    phases = np.angle(analytic)
    
    # Compute PLV
    plv = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        plv[i, i] = 1.0
        for j in range(i + 1, n_channels):
            # Phase difference
            phase_diff = phases[:, i, :] - phases[:, j, :]
            
            # PLV = |mean(exp(i * phase_diff))|
            plv_val = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            plv[i, j] = plv_val
            plv[j, i] = plv_val
    
    return plv


def compute_mutual_information(
    data: np.ndarray,
    n_bins: int = 32
) -> np.ndarray:
    """
    Compute mutual information between channels.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints)
    n_bins : int
        Number of bins for histogram estimation
        
    Returns
    -------
    mi : np.ndarray
        Mutual information matrix (n_channels, n_channels)
    """
    if data.ndim == 3:
        # Concatenate epochs
        n_epochs, n_channels, n_times = data.shape
        data = data.transpose(1, 0, 2).reshape(n_channels, -1)
    
    n_channels = data.shape[0]
    mi_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                # Self MI = entropy
                hist, _ = np.histogram(data[i], bins=n_bins, density=True)
                hist = hist[hist > 0]
                mi_matrix[i, i] = -np.sum(hist * np.log2(hist + 1e-10))
            else:
                mi_val = _mutual_info_2d(data[i], data[j], n_bins)
                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val
    
    return mi_matrix


def _mutual_info_2d(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute mutual information between two signals."""
    # Joint histogram
    joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    joint_hist = joint_hist + 1e-10  # Avoid log(0)
    
    # Marginal histograms
    px = np.sum(joint_hist, axis=1)
    py = np.sum(joint_hist, axis=0)
    
    # Normalize
    joint_hist = joint_hist / np.sum(joint_hist)
    px = px / np.sum(px)
    py = py / np.sum(py)
    
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint_hist[i, j] > 1e-10:
                mi += joint_hist[i, j] * np.log2(
                    joint_hist[i, j] / (px[i] * py[j] + 1e-10)
                )
    
    return max(0, mi)  # MI should be non-negative


def extract_connectivity_features(
    data: np.ndarray,
    sfreq: float,
    methods: Optional[list] = None,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive connectivity features.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    methods : list, optional
        Connectivity methods to compute
    freq_bands : dict, optional
        Frequency bands for band-specific connectivity
        
    Returns
    -------
    features : dict
        Dictionary of connectivity matrices
    """
    if methods is None:
        methods = ['correlation', 'coherence']
    
    if freq_bands is None:
        freq_bands = {
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 70)
        }
    
    features = {}
    
    if 'correlation' in methods:
        features['correlation'] = compute_correlation_matrix(data)
    
    if 'coherence' in methods:
        # Broadband coherence
        features['coherence'] = compute_coherence(data, sfreq)
        
        # Band-specific coherence
        for band_name, band in freq_bands.items():
            features[f'coherence_{band_name}'] = compute_coherence(
                data, sfreq, freq_band=band
            )
    
    if 'plv' in methods:
        for band_name, band in freq_bands.items():
            features[f'plv_{band_name}'] = compute_phase_locking_value(
                data, sfreq, freq_band=band
            )
    
    if 'mutual_information' in methods:
        features['mutual_information'] = compute_mutual_information(data)
    
    return features

