"""
Artifact Removal for SEEG Data

Methods for detecting and removing artifacts:
- Threshold-based detection
- ICA-based artifact removal
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from scipy import stats
import warnings


def detect_artifacts(
    data: np.ndarray,
    sfreq: float,
    method: str = 'threshold',
    threshold_std: float = 5.0,
    window_size: Optional[float] = None,
    return_mask: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Detect artifacts in SEEG data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints) or (n_epochs, n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    method : str
        Detection method: 'threshold', 'gradient', or 'both'
    threshold_std : float
        Threshold in standard deviations for artifact detection
    window_size : float, optional
        Window size in seconds for gradient detection
    return_mask : bool
        If True, return boolean mask; if False, return indices
        
    Returns
    -------
    artifact_mask : np.ndarray
        Boolean mask (True = artifact) or indices
    info : dict
        Information about detected artifacts (only if return_mask=True)
    """
    if data.ndim == 2:
        return _detect_artifacts_continuous(data, sfreq, method, threshold_std, 
                                            window_size, return_mask)
    elif data.ndim == 3:
        return _detect_artifacts_epoched(data, sfreq, method, threshold_std,
                                         window_size, return_mask)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def _detect_artifacts_continuous(
    data: np.ndarray,
    sfreq: float,
    method: str,
    threshold_std: float,
    window_size: Optional[float],
    return_mask: bool
) -> Tuple[np.ndarray, Dict]:
    """Detect artifacts in continuous data."""
    n_channels, n_timepoints = data.shape
    
    # Initialize mask
    artifact_mask = np.zeros(n_timepoints, dtype=bool)
    info = {'n_artifacts': 0, 'channels': [], 'method': method}
    
    if method in ['threshold', 'both']:
        # Threshold-based detection
        for ch_idx in range(n_channels):
            ch_data = data[ch_idx]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            threshold = threshold_std * std
            
            ch_artifacts = np.abs(ch_data - mean) > threshold
            artifact_mask |= ch_artifacts
            
            if np.any(ch_artifacts):
                info['channels'].append(ch_idx)
    
    if method in ['gradient', 'both']:
        # Gradient-based detection (for sudden jumps)
        for ch_idx in range(n_channels):
            ch_data = data[ch_idx]
            gradient = np.abs(np.diff(ch_data))
            grad_threshold = threshold_std * np.std(gradient)
            
            grad_artifacts = np.concatenate([[False], gradient > grad_threshold])
            artifact_mask |= grad_artifacts
    
    info['n_artifacts'] = np.sum(artifact_mask)
    info['artifact_ratio'] = info['n_artifacts'] / n_timepoints
    
    if return_mask:
        return artifact_mask, info
    else:
        return np.where(artifact_mask)[0], info


def _detect_artifacts_epoched(
    data: np.ndarray,
    sfreq: float,
    method: str,
    threshold_std: float,
    window_size: Optional[float],
    return_mask: bool
) -> Tuple[np.ndarray, Dict]:
    """Detect artifacts in epoched data."""
    n_epochs, n_channels, n_timepoints = data.shape
    
    # Mask per epoch
    epoch_mask = np.zeros(n_epochs, dtype=bool)
    info = {'n_bad_epochs': 0, 'bad_epochs': [], 'method': method}
    
    for ep_idx in range(n_epochs):
        ep_data = data[ep_idx]
        
        for ch_idx in range(n_channels):
            ch_data = ep_data[ch_idx]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            threshold = threshold_std * std
            
            if np.any(np.abs(ch_data - mean) > threshold):
                epoch_mask[ep_idx] = True
                info['bad_epochs'].append(ep_idx)
                break
    
    info['n_bad_epochs'] = np.sum(epoch_mask)
    info['bad_epoch_ratio'] = info['n_bad_epochs'] / n_epochs
    
    if return_mask:
        return epoch_mask, info
    else:
        return np.where(epoch_mask)[0], info


def remove_artifacts_threshold(
    data: np.ndarray,
    sfreq: float,
    threshold_std: float = 5.0,
    interpolate: bool = True,
    return_info: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Remove artifacts using threshold-based detection.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    threshold_std : float
        Threshold in standard deviations
    interpolate : bool
        If True, interpolate artifact segments; if False, set to NaN
    return_info : bool
        If True, also return info dict
        
    Returns
    -------
    cleaned_data : np.ndarray
        Data with artifacts removed/interpolated
    info : dict
        Information about artifacts (only if return_info=True)
    """
    artifact_mask, info = detect_artifacts(data, sfreq, method='threshold',
                                           threshold_std=threshold_std)
    
    cleaned_data = data.copy()
    
    if data.ndim == 2:
        # Continuous data: interpolate or set to NaN
        for ch_idx in range(data.shape[0]):
            if interpolate:
                cleaned_data[ch_idx] = _interpolate_artifacts(
                    cleaned_data[ch_idx], artifact_mask
                )
            else:
                cleaned_data[ch_idx, artifact_mask] = np.nan
    else:
        # Epoched data: remove bad epochs
        good_epochs = ~artifact_mask
        cleaned_data = cleaned_data[good_epochs]
        info['n_epochs_remaining'] = np.sum(good_epochs)
    
    if return_info:
        return cleaned_data, info
    return cleaned_data


def _interpolate_artifacts(data_1d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate artifact segments in 1D data."""
    cleaned = data_1d.copy()
    good_indices = np.where(~mask)[0]
    bad_indices = np.where(mask)[0]
    
    if len(good_indices) > 1 and len(bad_indices) > 0:
        cleaned[bad_indices] = np.interp(bad_indices, good_indices, 
                                         cleaned[good_indices])
    
    return cleaned


def remove_artifacts_ica(
    data: np.ndarray,
    sfreq: float,
    n_components: Optional[int] = None,
    exclude_components: Optional[List[int]] = None,
    threshold: float = 3.0,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Remove artifacts using ICA.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    n_components : int, optional
        Number of ICA components (default: n_channels)
    exclude_components : list of int, optional
        Indices of components to remove (if known)
    threshold : float
        Kurtosis threshold for automatic artifact component detection
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    cleaned_data : np.ndarray
        Data with artifact components removed
    info : dict
        Information about ICA and removed components
    """
    from sklearn.decomposition import FastICA
    
    if data.ndim != 2:
        raise ValueError("ICA artifact removal requires 2D data (n_channels, n_timepoints)")
    
    n_channels, n_timepoints = data.shape
    
    if n_components is None:
        n_components = min(n_channels, 50)  # Cap at 50 for efficiency
    
    # Store the mean for reconstruction (FastICA centers the data internally)
    data_mean = np.mean(data, axis=1, keepdims=True)
    
    # Fit ICA
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=500)
    
    # ICA expects (n_samples, n_features), so transpose
    sources = ica.fit_transform(data.T).T  # Shape: (n_components, n_timepoints)
    mixing_matrix = ica.mixing_  # Shape: (n_channels, n_components)
    
    info = {
        'n_components': n_components,
        'excluded_components': [],
        'component_kurtosis': []
    }
    
    # Identify artifact components
    if exclude_components is None:
        exclude_components = []
        for i in range(n_components):
            # Use kurtosis to identify artifact components
            # (artifacts often have high kurtosis)
            kurtosis = stats.kurtosis(sources[i])
            info['component_kurtosis'].append(kurtosis)
            
            if np.abs(kurtosis) > threshold:
                exclude_components.append(i)
    
    info['excluded_components'] = exclude_components
    
    # Remove artifact components
    sources_cleaned = sources.copy()
    sources_cleaned[exclude_components] = 0
    
    # Reconstruct data and restore the mean
    cleaned_data = (mixing_matrix @ sources_cleaned) + data_mean
    
    return cleaned_data, info
