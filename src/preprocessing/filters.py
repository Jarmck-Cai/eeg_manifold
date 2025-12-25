"""
Filtering Functions for SEEG Data

Provides bandpass, highpass, lowpass, and notch filtering.
"""

import numpy as np
from scipy import signal
from typing import Optional, Union, Tuple
import warnings


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply bandpass Butterworth filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency in Hz
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    order : int
        Filter order (default: 4)
    axis : int
        Axis along which to filter (default: -1, last axis)
        
    Returns
    -------
    np.ndarray
        Filtered data
        
    Examples
    --------
    >>> filtered = bandpass_filter(data, sfreq=1000, lowcut=1, highcut=150)
    """
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    
    # Check for valid frequency range
    if low <= 0:
        warnings.warn(f"lowcut ({lowcut}) is too low, using 0.1 Hz")
        low = 0.1 / nyq
    if high >= 1:
        warnings.warn(f"highcut ({highcut}) is >= Nyquist, using {nyq - 1} Hz")
        high = (nyq - 1) / nyq
    
    # Design filter
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    # Apply filter (forward-backward for zero phase)
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    
    return filtered


def highpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply highpass Butterworth filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency in Hz
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order
    axis : int
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        Filtered data
    """
    nyq = 0.5 * sfreq
    normalized_cutoff = cutoff / nyq
    
    if normalized_cutoff <= 0:
        warnings.warn(f"cutoff ({cutoff}) is too low, using 0.1 Hz")
        normalized_cutoff = 0.1 / nyq
    
    sos = signal.butter(order, normalized_cutoff, btype='high', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    
    return filtered


def lowpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff: float,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply lowpass Butterworth filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency in Hz
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order
    axis : int
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        Filtered data
    """
    nyq = 0.5 * sfreq
    normalized_cutoff = cutoff / nyq
    
    if normalized_cutoff >= 1:
        warnings.warn(f"cutoff ({cutoff}) >= Nyquist, using {nyq - 1} Hz")
        normalized_cutoff = (nyq - 1) / nyq
    
    sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    
    return filtered


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    width: float = 2.0,
    harmonics: bool = True,
    axis: int = -1
) -> np.ndarray:
    """
    Apply notch filter to remove line noise.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency in Hz
    freq : float
        Frequency to notch out (e.g., 50 or 60 Hz)
    width : float
        Width of notch in Hz (default: 2.0)
    harmonics : bool
        Whether to also filter harmonics (default: True)
    axis : int
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        Filtered data
        
    Examples
    --------
    >>> # Remove 50 Hz line noise and harmonics
    >>> filtered = notch_filter(data, sfreq=1000, freq=50, harmonics=True)
    """
    nyq = 0.5 * sfreq
    filtered = data.copy()
    
    # Calculate quality factor
    Q = freq / width
    
    # Frequencies to notch (including harmonics if requested)
    if harmonics:
        notch_freqs = []
        f = freq
        while f < nyq:
            notch_freqs.append(f)
            f += freq
    else:
        notch_freqs = [freq]
    
    # Apply notch at each frequency
    for nf in notch_freqs:
        if nf >= nyq:
            break
        b, a = signal.iirnotch(nf / nyq, Q)
        filtered = signal.filtfilt(b, a, filtered, axis=axis)
    
    return filtered


def filter_data(
    data: np.ndarray,
    sfreq: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    notch_freq: Optional[float] = None,
    notch_width: float = 2.0,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Comprehensive filtering function combining bandpass and notch.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency in Hz
    lowcut : float, optional
        Lower cutoff for bandpass (highpass if highcut is None)
    highcut : float, optional
        Upper cutoff for bandpass (lowpass if lowcut is None)
    notch_freq : float, optional
        Notch filter frequency (e.g., 50 or 60 for line noise)
    notch_width : float
        Width of notch filter in Hz
    order : int
        Filter order for Butterworth filter
    axis : int
        Axis along which to filter
        
    Returns
    -------
    np.ndarray
        Filtered data
        
    Examples
    --------
    >>> # Complete preprocessing filter
    >>> filtered = filter_data(data, sfreq=1000, 
    ...                        lowcut=1, highcut=150, 
    ...                        notch_freq=50)
    """
    filtered = data.copy()
    
    # Apply bandpass/highpass/lowpass
    if lowcut is not None and highcut is not None:
        filtered = bandpass_filter(filtered, sfreq, lowcut, highcut, order, axis)
    elif lowcut is not None:
        filtered = highpass_filter(filtered, sfreq, lowcut, order, axis)
    elif highcut is not None:
        filtered = lowpass_filter(filtered, sfreq, highcut, order, axis)
    
    # Apply notch filter
    if notch_freq is not None:
        filtered = notch_filter(filtered, sfreq, notch_freq, notch_width, 
                               harmonics=True, axis=axis)
    
    return filtered
