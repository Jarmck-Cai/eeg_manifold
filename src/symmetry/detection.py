"""
Symmetry Detection

Methods for detecting symmetries and periodic structures in manifold embeddings.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from scipy import stats
from scipy.fft import fft, fftfreq


def detect_rotational_symmetry(
    embedding: np.ndarray,
    n_folds: Optional[List[int]] = None,
    center: Optional[np.ndarray] = None,
    n_bins: int = 360
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Detect rotational symmetry in 2D or 3D embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding (n_points, 2 or 3)
    n_folds : list of int, optional
        Fold numbers to test (default: 2-8)
    center : np.ndarray, optional
        Center point (default: centroid)
    n_bins : int
        Number of angular bins
        
    Returns
    -------
    result : dict
        Symmetry detection results including:
        - best_fold: most likely rotational symmetry
        - fold_scores: scores for each tested fold
        - angular_histogram: histogram of angular distribution
    """
    if n_folds is None:
        n_folds = [2, 3, 4, 5, 6, 7, 8]
    
    n_dim = embedding.shape[1]
    
    if center is None:
        center = np.mean(embedding, axis=0)
    
    # Center the data
    centered = embedding - center
    
    if n_dim == 2:
        # Compute angles
        angles = np.arctan2(centered[:, 1], centered[:, 0])
    elif n_dim >= 3:
        # Use first two PCs for 2D angle analysis
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        centered_2d = pca.fit_transform(centered)
        angles = np.arctan2(centered_2d[:, 1], centered_2d[:, 0])
    else:
        raise ValueError(f"Expected 2D or 3D embedding, got {n_dim}D")
    
    # Convert to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)
    
    # Create histogram
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
    hist = hist / np.sum(hist)  # Normalize
    
    # Test each fold number using Fourier analysis
    fft_result = fft(hist)
    power = np.abs(fft_result) ** 2
    
    fold_scores = {}
    for fold in n_folds:
        if fold < len(power):
            fold_scores[fold] = power[fold]
    
    # Find best fold
    if fold_scores:
        best_fold = max(fold_scores, key=fold_scores.get)
        best_score = fold_scores[best_fold]
        
        # Normalize score by DC component
        if power[0] > 0:
            normalized_scores = {k: v / power[0] for k, v in fold_scores.items()}
        else:
            normalized_scores = fold_scores
    else:
        best_fold = 1
        best_score = 0
        normalized_scores = {}
    
    return {
        'best_fold': best_fold,
        'best_score': best_score,
        'fold_scores': fold_scores,
        'normalized_scores': normalized_scores,
        'angular_histogram': hist,
        'power_spectrum': power[:len(n_folds) + 2]
    }


def detect_periodic_structure(
    trajectory: np.ndarray,
    method: str = 'autocorr'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Detect periodic structure in trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Time series or trajectory (n_points, n_dim)
    method : str
        Detection method: 'autocorr' or 'fft'
        
    Returns
    -------
    result : dict
        Period detection results
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_points, n_dim = trajectory.shape
    
    if method == 'autocorr':
        # Compute autocorrelation for each dimension
        max_lag = n_points // 2
        autocorrs = np.zeros((n_dim, max_lag))
        
        for d in range(n_dim):
            sig = trajectory[:, d]
            sig = sig - np.mean(sig)
            
            for lag in range(max_lag):
                autocorrs[d, lag] = np.corrcoef(
                    sig[:n_points - lag], sig[lag:]
                )[0, 1]
        
        # Average across dimensions
        mean_autocorr = np.mean(autocorrs, axis=0)
        
        # Find first peak after initial decay
        # Skip the first few lags where autocorr is high
        start_idx = max(1, n_points // 20)
        peaks = []
        
        for i in range(start_idx, max_lag - 1):
            if mean_autocorr[i-1] < mean_autocorr[i] > mean_autocorr[i+1]:
                if mean_autocorr[i] > 0.1:  # Threshold
                    peaks.append((i, mean_autocorr[i]))
        
        if peaks:
            # Take the first significant peak as the period
            period = peaks[0][0]
            strength = peaks[0][1]
        else:
            period = 0
            strength = 0
        
        return {
            'period': period,
            'strength': strength,
            'autocorrelation': mean_autocorr,
            'peaks': peaks
        }
    
    elif method == 'fft':
        # FFT-based period detection
        spectra = []
        
        for d in range(n_dim):
            sig = trajectory[:, d]
            sig = sig - np.mean(sig)
            
            fft_result = fft(sig)
            power = np.abs(fft_result[:n_points // 2]) ** 2
            spectra.append(power)
        
        mean_spectrum = np.mean(spectra, axis=0)
        freqs = fftfreq(n_points)[:n_points // 2]
        
        # Find dominant frequency (excluding DC)
        peak_idx = np.argmax(mean_spectrum[1:]) + 1
        dominant_freq = freqs[peak_idx]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
        else:
            period = 0
        
        return {
            'period': period,
            'dominant_frequency': dominant_freq,
            'power_spectrum': mean_spectrum,
            'frequencies': freqs
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def test_translation_invariance(
    data: np.ndarray,
    shift: int,
    metric: str = 'correlation'
) -> float:
    """
    Test for translation invariance in time series.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_times)
    shift : int
        Time shift to test
    metric : str
        Similarity metric
        
    Returns
    -------
    similarity : float
        Similarity between original and shifted data
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_times = data.shape
    
    if shift >= n_times:
        raise ValueError(f"Shift {shift} >= data length {n_times}")
    
    # Compare original and shifted
    original = data[:, :n_times - shift]
    shifted = data[:, shift:]
    
    if metric == 'correlation':
        similarities = []
        for ch in range(n_channels):
            corr = np.corrcoef(original[ch], shifted[ch])[0, 1]
            similarities.append(corr)
        return np.mean(similarities)
    
    elif metric == 'mse':
        mse = np.mean((original - shifted) ** 2)
        # Convert to similarity
        return np.exp(-mse)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def detect_reflection_symmetry(
    embedding: np.ndarray,
    n_angles: int = 180
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Detect reflection symmetry in 2D embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding (n_points, 2)
    n_angles : int
        Number of reflection axes to test
        
    Returns
    -------
    result : dict
        Symmetry detection results
    """
    if embedding.shape[1] != 2:
        raise ValueError("Reflection symmetry detection requires 2D embedding")
    
    # Center the data
    centered = embedding - np.mean(embedding, axis=0)
    
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    symmetry_scores = np.zeros(n_angles)
    
    for i, angle in enumerate(angles):
        # Create reflection matrix
        cos2 = np.cos(2 * angle)
        sin2 = np.sin(2 * angle)
        reflection = np.array([[cos2, sin2], [sin2, -cos2]])
        
        # Reflect points
        reflected = centered @ reflection.T
        
        # Compute similarity (how close reflected points are to original set)
        from scipy.spatial.distance import cdist
        distances = cdist(reflected, centered)
        min_distances = np.min(distances, axis=1)
        
        # Score: inverse of mean minimum distance
        symmetry_scores[i] = 1 / (np.mean(min_distances) + 1e-10)
    
    # Normalize
    symmetry_scores = symmetry_scores / np.max(symmetry_scores)
    
    best_idx = np.argmax(symmetry_scores)
    best_angle = angles[best_idx]
    best_score = symmetry_scores[best_idx]
    
    return {
        'best_angle': best_angle,
        'best_angle_degrees': np.degrees(best_angle),
        'best_score': best_score,
        'all_scores': symmetry_scores,
        'angles': angles
    }

