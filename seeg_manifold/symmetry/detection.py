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
    n_bins: int = 360,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Detect rotational symmetry in a 2D or 3D embedding.

    The angular distribution of the points is binned and Fourier
    transformed; ``k``-fold rotational structure appears as power at
    harmonic ``k``. Because the strongest harmonic is always *some*
    harmonic, raw fold scores cannot distinguish structure from noise.
    Each fold is therefore tested against a null distribution built from
    Gaussian surrogates matched to the covariance of the analysed plane,
    and a permutation p-value is reported alongside the score. Matching
    the covariance matters: any elongated cloud has a non-uniform angular
    distribution, so a uniform-angle null would flag ordinary elliptical
    noise as symmetric.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding (n_points, 2 or 3). For 3+ dimensions the angles are
        computed in the plane of the first two principal components.
    n_folds : list of int, optional
        Fold numbers to test (default: 2-8)
    center : np.ndarray, optional
        Center point (default: centroid)
    n_bins : int
        Number of angular bins
    n_permutations : int
        Number of covariance-matched Gaussian surrogates used for the null
        distribution. Set to 0 to skip the test (p-values become NaN).
    random_state : int, optional
        Seed for the surrogate draws, for reproducibility.

    Returns
    -------
    result : dict
        Symmetry detection results including:

        - ``best_fold``: fold with the highest normalized score
        - ``best_score``: its DC-normalized spectral power
        - ``best_p_value``: permutation p-value for ``best_fold``,
          corrected across the tested folds (Bonferroni)
        - ``significant``: whether ``best_p_value`` < 0.05
        - ``fold_scores`` / ``normalized_scores``: per-fold scores
        - ``p_values``: uncorrected per-fold permutation p-values
        - ``angular_histogram``: histogram of the angular distribution

    Notes
    -----
    A p-value near 1 means the angular distribution is indistinguishable
    from a Gaussian of the same shape at that fold. Report ``best_fold`` only when
    ``significant`` is True.
    """
    if n_folds is None:
        n_folds = [2, 3, 4, 5, 6, 7, 8]
    
    n_dim = embedding.shape[1]
    
    if center is None:
        center = np.mean(embedding, axis=0)
    
    # Center the data
    centered = embedding - center
    
    if n_dim == 2:
        plane = centered
    elif n_dim >= 3:
        # Use first two PCs for 2D angle analysis
        from sklearn.decomposition import PCA
        plane = PCA(n_components=2).fit_transform(centered)
    else:
        raise ValueError(f"Expected 2D or 3D embedding, got {n_dim}D")

    angles = np.arctan2(plane[:, 1], plane[:, 0])
    
    # Convert to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)

    # Angular histogram and its harmonic power spectrum
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
    hist = hist / np.sum(hist)
    power = _angular_fold_power(angles, n_bins)

    tested_folds = [f for f in n_folds if f < len(power)]
    fold_scores = {f: float(power[f]) for f in tested_folds}
    if power[0] > 0:
        normalized_scores = {f: float(power[f] / power[0]) for f in tested_folds}
    else:
        normalized_scores = dict(fold_scores)

    if not tested_folds:
        return {
            'best_fold': 1, 'best_score': 0.0, 'best_p_value': float('nan'),
            'significant': False, 'fold_scores': {}, 'normalized_scores': {},
            'p_values': {}, 'n_permutations': 0,
            'angular_histogram': hist, 'power_spectrum': power[:2],
        }

    best_fold = max(normalized_scores, key=normalized_scores.get)
    best_score = normalized_scores[best_fold]

    # Null distribution: Gaussian surrogates matched to the covariance of
    # the analysed plane.
    #
    # A uniform-angle null would be wrong here. Any anisotropic cloud has a
    # non-uniform angular distribution, and PCA guarantees anisotropy
    # (PC1 carries more variance than PC2), so a uniform null flags plain
    # elliptical noise as symmetric. Matching the surrogate covariance to
    # the data tests for angular structure *beyond* second-order shape,
    # which is the question actually being asked.
    p_values = {f: float('nan') for f in tested_folds}
    best_p = float('nan')
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        n_points = len(angles)
        cov = np.cov(plane, rowvar=False)
        exceed = {f: 0 for f in tested_folds}
        for _ in range(n_permutations):
            surrogate = rng.multivariate_normal(np.zeros(2), cov, size=n_points)
            null_angles = np.mod(np.arctan2(surrogate[:, 1], surrogate[:, 0]),
                                 2 * np.pi)
            null_power = _angular_fold_power(null_angles, n_bins)
            dc = null_power[0] if null_power[0] > 0 else 1.0
            for f in tested_folds:
                if null_power[f] / dc >= normalized_scores[f]:
                    exceed[f] += 1
        # Add-one correction keeps the p-value strictly positive.
        p_values = {f: (exceed[f] + 1) / (n_permutations + 1) for f in tested_folds}
        # Bonferroni-correct the winner for having scanned several folds.
        best_p = min(1.0, p_values[best_fold] * len(tested_folds))

    return {
        'best_fold': best_fold,
        'best_score': best_score,
        'best_p_value': best_p,
        'significant': bool(best_p < 0.05) if best_p == best_p else False,
        'fold_scores': fold_scores,
        'normalized_scores': normalized_scores,
        'p_values': p_values,
        'n_permutations': n_permutations,
        'angular_histogram': hist,
        'power_spectrum': power[:len(n_folds) + 2]
    }


def _angular_fold_power(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Spectral power of the angular histogram, per rotational harmonic."""
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
    hist = hist / np.sum(hist)
    return np.abs(fft(hist)) ** 2


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
    n_angles: int = 180,
    n_permutations: int = 200,
    random_state: Optional[int] = None,
    max_points: Optional[int] = 1000
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Detect reflection symmetry in a 2D embedding.

    For each candidate axis the point cloud is reflected and scored by
    how well the reflected points fall onto the original ones (the
    negative mean nearest-neighbour distance). The best axis is then
    tested against surrogates that keep each point's radius but
    randomise its angle, which destroys reflection structure while
    preserving the radial profile.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding (n_points, 2)
    n_angles : int
        Number of reflection axes to test
    n_permutations : int
        Number of surrogates for the null distribution. Set to 0 to skip
        the test (the p-value becomes NaN).
    random_state : int, optional
        Seed for the surrogate draws, for reproducibility.
    max_points : int, optional
        Subsample the cloud to at most this many points before testing.
        The cost is O(n_permutations x n_angles x n log n), so large
        embeddings are impractical without this. Set to None to disable.

    Returns
    -------
    result : dict
        Symmetry detection results including ``best_angle``,
        ``best_score`` (mean nearest-neighbour residual, lower is more
        symmetric), ``best_p_value`` and ``significant``.

    Notes
    -----
    ``best_score`` is an absolute residual in embedding units, not a
    normalised quantity: scores are comparable across axes of the same
    cloud but not across different clouds. Use ``best_p_value`` to judge
    whether any reflection symmetry is present at all.
    """
    if embedding.shape[1] != 2:
        raise ValueError("Reflection symmetry detection requires 2D embedding")

    # Center the data
    centered = embedding - np.mean(embedding, axis=0)

    # The permutation test evaluates every axis for every surrogate, so
    # subsample large clouds to keep the runtime bounded. The residual is
    # a mean over points and is stable under subsampling.
    if max_points is not None and len(centered) > max_points:
        rng_sub = np.random.default_rng(random_state)
        centered = centered[rng_sub.choice(len(centered), max_points, replace=False)]

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    residuals = _reflection_residuals(centered, angles)

    best_idx = int(np.argmin(residuals))
    best_angle = angles[best_idx]
    best_score = float(residuals[best_idx])

    # Null: Gaussian surrogates matched to the covariance of the cloud.
    # An elongated cloud is already reflection-symmetric about its
    # principal axes, so the meaningful question is whether the data are
    # *more* reflection-symmetric than their own second-order shape
    # implies. Randomising angles at fixed radius would not control for
    # that and would report elliptical noise as symmetric.
    best_p = float('nan')
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        cov = np.cov(centered, rowvar=False)
        exceed = 0
        for _ in range(n_permutations):
            surrogate = rng.multivariate_normal(np.zeros(2), cov, size=len(centered))
            # More symmetric == smaller residual, so count surrogates
            # that are at least as symmetric as the observed cloud.
            if np.min(_reflection_residuals(surrogate, angles)) <= best_score:
                exceed += 1
        best_p = (exceed + 1) / (n_permutations + 1)

    return {
        'best_angle': best_angle,
        'best_angle_degrees': np.degrees(best_angle),
        'best_score': best_score,
        'best_p_value': best_p,
        'significant': bool(best_p < 0.05) if best_p == best_p else False,
        'n_permutations': n_permutations,
        'all_scores': residuals,
        'angles': angles
    }


def _reflection_residuals(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Mean nearest-neighbour distance after reflecting across each axis.

    Uses a KD-tree rather than a full pairwise distance matrix: the
    permutation test evaluates this for every surrogate and every
    candidate axis, so the O(n^2) form is not affordable.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    residuals = np.zeros(len(angles))
    for i, angle in enumerate(angles):
        cos2, sin2 = np.cos(2 * angle), np.sin(2 * angle)
        reflection = np.array([[cos2, sin2], [sin2, -cos2]])
        reflected = points @ reflection.T
        distances, _ = tree.query(reflected, k=1)
        residuals[i] = np.mean(distances)
    return residuals

