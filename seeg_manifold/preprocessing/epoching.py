"""
Epoching Functions for SEEG Data

Create epochs from continuous data for analysis.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import warnings


def create_epochs(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float,
    overlap: float = 0.0,
    events: Optional[np.ndarray] = None,
    tmin: float = 0.0,
    tmax: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create epochs from continuous data.
    
    Parameters
    ----------
    data : np.ndarray
        Continuous data (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency in Hz
    epoch_length : float
        Length of each epoch in seconds
    overlap : float
        Overlap between epochs (0-1, default: 0)
    events : np.ndarray, optional
        Event times for event-locked epoching
    tmin : float
        Start time relative to event (for event-locked)
    tmax : float, optional
        End time relative to event (for event-locked)
        
    Returns
    -------
    epochs : np.ndarray
        Epoched data (n_epochs, n_channels, n_timepoints)
    times : np.ndarray
        Time vector for each epoch
        
    Examples
    --------
    >>> # Create 2-second epochs with 50% overlap
    >>> epochs, times = create_epochs(data, sfreq=1000, 
    ...                               epoch_length=2.0, overlap=0.5)
    
    >>> # Event-locked epoching
    >>> epochs, times = create_epochs(data, sfreq=1000,
    ...                               events=event_times, tmin=-0.5, tmax=1.5)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    
    n_channels, n_timepoints = data.shape

    if events is not None:
        if tmax is None:
            raise ValueError(
                "tmax is required for event-locked epoching; it sets how far "
                "each epoch extends after the event (e.g. tmin=-0.2, tmax=0.8)."
            )
        return _epoch_event_locked(data, sfreq, events, tmin, tmax)
    else:
        return _epoch_fixed_length(data, sfreq, epoch_length, overlap)


def _epoch_fixed_length(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float,
    overlap: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed-length epochs."""
    n_channels, n_timepoints = data.shape
    
    # Calculate sample parameters
    samples_per_epoch = int(epoch_length * sfreq)
    step = int(samples_per_epoch * (1 - overlap))
    
    if step <= 0:
        raise ValueError(f"Overlap {overlap} is too high, step size would be <= 0")
    
    # Calculate number of epochs
    n_epochs = (n_timepoints - samples_per_epoch) // step + 1
    
    if n_epochs <= 0:
        raise ValueError(f"Data too short for epoch_length={epoch_length}s")
    
    # Create epochs
    epochs = np.zeros((n_epochs, n_channels, samples_per_epoch))
    
    for i in range(n_epochs):
        start = i * step
        end = start + samples_per_epoch
        epochs[i] = data[:, start:end]
    
    # Create time vector
    times = np.arange(samples_per_epoch) / sfreq
    
    return epochs, times


def _epoch_event_locked(
    data: np.ndarray,
    sfreq: float,
    events: np.ndarray,
    tmin: float,
    tmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Create event-locked epochs."""
    n_channels, n_timepoints = data.shape
    
    # Convert times to samples using round for better accuracy
    samples_before = int(round(abs(tmin) * sfreq))
    samples_after = int(round(tmax * sfreq))
    samples_per_epoch = samples_before + samples_after
    
    # Filter valid events
    valid_events = []
    for event in events:
        event_sample = int(event * sfreq) if isinstance(event, float) else int(event)
        start = event_sample - samples_before
        end = event_sample + samples_after
        
        if start >= 0 and end <= n_timepoints:
            valid_events.append(event_sample)
    
    if len(valid_events) == 0:
        raise ValueError("No valid events found within data bounds")
    
    n_epochs = len(valid_events)
    
    # Create epochs
    epochs = np.zeros((n_epochs, n_channels, samples_per_epoch))
    
    for i, event_sample in enumerate(valid_events):
        start = event_sample - samples_before
        end = event_sample + samples_after
        epochs[i] = data[:, start:end]
    
    # Create time vector based on actual sample indices
    # First sample is at -samples_before/sfreq, each step is 1/sfreq
    times = np.arange(samples_per_epoch) / sfreq - samples_before / sfreq
    
    return epochs, times


def epoch_data(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 2.0,
    overlap: float = 0.5,
    reject_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Epoch data with optional artifact rejection.
    
    This is a convenience function that combines epoching with
    basic artifact rejection.
    
    Parameters
    ----------
    data : np.ndarray
        Continuous data (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    epoch_length : float
        Epoch length in seconds
    overlap : float
        Overlap ratio (0-1)
    reject_threshold : float, optional
        Threshold in standard deviations for epoch rejection
        
    Returns
    -------
    epochs : np.ndarray
        Epoched data
    times : np.ndarray
        Time vector
    info : dict
        Epoching information
    """
    # Create epochs
    epochs, times = create_epochs(data, sfreq, epoch_length, overlap)
    
    info = {
        'n_epochs_original': epochs.shape[0],
        'epoch_length': epoch_length,
        'overlap': overlap,
        'sfreq': sfreq
    }
    
    # Optional rejection
    if reject_threshold is not None:
        # Calculate epoch-wise max amplitude
        epoch_max = np.max(np.abs(epochs), axis=(1, 2))
        # Reject epochs whose peak amplitude is an outlier relative to the
        # rest of the recording, using a median/MAD criterion.
        #
        # Two failure modes motivate this form. A bare
        # ``reject_threshold * std`` (no location term) compares an
        # absolute amplitude against a spread and rejects *every* epoch
        # whenever the peak amplitudes are tightly clustered, which is
        # the normal case for clean data. Using mean and std instead
        # fixes that but breaks the opposite case: a single large
        # artifact inflates both statistics enough to mask itself.
        # Median and MAD are robust to both.
        center = np.median(epoch_max)
        mad = np.median(np.abs(epoch_max - center))
        # 1.4826 rescales the MAD to a standard-deviation equivalent for
        # normally distributed data, so reject_threshold keeps its
        # "number of sigmas" meaning.
        scale = 1.4826 * mad
        if scale <= 0:
            # Degenerate spread (e.g. identical epochs): fall back to the
            # standard deviation, and keep everything if that is zero too.
            scale = np.std(epoch_max)
        if scale <= 0:
            good_epochs = np.ones(len(epoch_max), dtype=bool)
        else:
            good_epochs = epoch_max < center + reject_threshold * scale
        
        epochs = epochs[good_epochs]
        info['n_epochs_rejected'] = np.sum(~good_epochs)
        info['n_epochs_final'] = epochs.shape[0]
    else:
        info['n_epochs_final'] = epochs.shape[0]
        info['n_epochs_rejected'] = 0
    
    return epochs, times, info


def concatenate_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Concatenate epochs back into continuous data.
    
    Parameters
    ----------
    epochs : np.ndarray
        Epoched data (n_epochs, n_channels, n_timepoints)
        
    Returns
    -------
    np.ndarray
        Continuous data (n_channels, n_epochs * n_timepoints)
    """
    if epochs.ndim != 3:
        raise ValueError(f"Expected 3D array, got {epochs.ndim}D")
    
    n_epochs, n_channels, n_timepoints = epochs.shape
    
    # Reshape: (n_epochs, n_channels, n_timepoints) -> (n_channels, n_epochs * n_timepoints)
    return epochs.transpose(1, 0, 2).reshape(n_channels, -1)
