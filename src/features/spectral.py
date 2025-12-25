"""
Spectral Feature Extraction

Extract frequency-domain features from SEEG data.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from scipy import signal


# Default frequency bands
DEFAULT_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'low_gamma': (30, 70),
    'high_gamma': (70, 150)
}


def compute_psd(
    data: np.ndarray,
    sfreq: float,
    method: str = 'welch',
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints) or (n_epochs, n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    method : str
        PSD method: 'welch' or 'multitaper'
    nperseg : int, optional
        Segment length for Welch method
    noverlap : int, optional
        Overlap for Welch method
    nfft : int, optional
        FFT length
        
    Returns
    -------
    freqs : np.ndarray
        Frequency vector
    psd : np.ndarray
        Power spectral density
    """
    if nperseg is None:
        nperseg = min(256, data.shape[-1])
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    if method == 'welch':
        if data.ndim == 2:
            freqs, psd = signal.welch(
                data, fs=sfreq, nperseg=nperseg, 
                noverlap=noverlap, nfft=nfft, axis=-1
            )
        elif data.ndim == 3:
            # Average across epochs
            freqs, psd = signal.welch(
                data, fs=sfreq, nperseg=nperseg,
                noverlap=noverlap, nfft=nfft, axis=-1
            )
            psd = np.mean(psd, axis=0)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    elif method == 'multitaper':
        try:
            from scipy.signal.windows import dpss
        except ImportError:
            from scipy.signal import windows
            dpss = windows.dpss
        
        # Use DPSS windows for multitaper
        NW = 4  # Time-bandwidth product
        Kmax = int(2 * NW - 1)
        
        n_times = data.shape[-1]
        tapers = dpss(n_times, NW, Kmax)
        
        if data.ndim == 2:
            n_channels = data.shape[0]
            # Apply each taper
            psds = []
            for taper in tapers:
                tapered = data * taper
                freqs, psd_taper = signal.periodogram(
                    tapered, fs=sfreq, nfft=nfft, axis=-1
                )
                psds.append(psd_taper)
            psd = np.mean(psds, axis=0)
        else:
            # For epoched data
            n_epochs, n_channels, _ = data.shape
            psds = []
            for taper in tapers:
                tapered = data * taper
                freqs, psd_taper = signal.periodogram(
                    tapered, fs=sfreq, nfft=nfft, axis=-1
                )
                psds.append(psd_taper)
            psd = np.mean(psds, axis=0)
            psd = np.mean(psd, axis=0)  # Average across epochs
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return freqs, psd


def compute_band_power(
    data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'welch',
    normalize: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute power in frequency bands.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    bands : dict, optional
        Frequency bands as {name: (low, high)}
    method : str
        PSD method
    normalize : bool
        If True, return relative band power
        
    Returns
    -------
    band_power : dict
        Dictionary mapping band names to power arrays
    """
    if bands is None:
        bands = DEFAULT_BANDS
    
    freqs, psd = compute_psd(data, sfreq, method=method, **kwargs)
    
    band_power = {}
    
    for band_name, (low, high) in bands.items():
        # Find frequency indices in band
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        if psd.ndim == 1:
            power = np.trapz(psd[idx], freqs[idx])
        else:
            power = np.trapz(psd[:, idx], freqs[idx], axis=-1)
        
        band_power[band_name] = power
    
    if normalize:
        # Compute total power
        if psd.ndim == 1:
            total_power = np.trapz(psd, freqs)
        else:
            total_power = np.trapz(psd, freqs, axis=-1)
        
        for band_name in band_power:
            band_power[band_name] = band_power[band_name] / total_power
    
    return band_power


def compute_spectral_entropy(
    data: np.ndarray,
    sfreq: float,
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Compute spectral entropy.
    
    Spectral entropy measures the complexity/regularity of the signal.
    Low entropy = regular/periodic, high entropy = irregular/noisy.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    normalize : bool
        If True, normalize by log(n_freqs)
        
    Returns
    -------
    entropy : np.ndarray
        Spectral entropy per channel
    """
    freqs, psd = compute_psd(data, sfreq, **kwargs)
    
    # Normalize PSD to probability distribution
    if psd.ndim == 1:
        psd_norm = psd / np.sum(psd)
        # Avoid log(0)
        psd_norm = psd_norm + 1e-10
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        if normalize:
            entropy = entropy / np.log2(len(freqs))
    else:
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        psd_norm = psd_norm + 1e-10
        entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=-1)
        
        if normalize:
            entropy = entropy / np.log2(len(freqs))
    
    return entropy


def compute_peak_frequency(
    data: np.ndarray,
    sfreq: float,
    freq_range: Optional[Tuple[float, float]] = None,
    **kwargs
) -> np.ndarray:
    """
    Find peak frequency in PSD.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    freq_range : tuple, optional
        Frequency range to search (low, high)
        
    Returns
    -------
    peak_freq : np.ndarray
        Peak frequency per channel
    """
    freqs, psd = compute_psd(data, sfreq, **kwargs)
    
    if freq_range is not None:
        idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
        freqs = freqs[idx]
        if psd.ndim == 1:
            psd = psd[idx]
        else:
            psd = psd[:, idx]
    
    if psd.ndim == 1:
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
    else:
        peak_idx = np.argmax(psd, axis=-1)
        peak_freq = freqs[peak_idx]
    
    return peak_freq


def extract_spectral_features(
    data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    include_entropy: bool = True,
    include_peak_freq: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive spectral features.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    sfreq : float
        Sampling frequency
    bands : dict, optional
        Frequency bands
    include_entropy : bool
        Whether to include spectral entropy
    include_peak_freq : bool
        Whether to include peak frequency
        
    Returns
    -------
    features : dict
        Dictionary of spectral features
    """
    features = {}
    
    # Band power
    band_power = compute_band_power(data, sfreq, bands=bands, normalize=True, **kwargs)
    for band_name, power in band_power.items():
        features[f'power_{band_name}'] = power
    
    # Spectral entropy
    if include_entropy:
        features['spectral_entropy'] = compute_spectral_entropy(data, sfreq, **kwargs)
    
    # Peak frequency
    if include_peak_freq:
        features['peak_frequency'] = compute_peak_frequency(data, sfreq, **kwargs)
        # Peak frequency in alpha band
        features['peak_frequency_alpha'] = compute_peak_frequency(
            data, sfreq, freq_range=(8, 13), **kwargs
        )
    
    return features

