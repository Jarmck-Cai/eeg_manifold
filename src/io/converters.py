"""
Data Format Converters

Utilities for converting between different data formats:
- MATLAB to MNE-Python
- MNE-Python to NumPy arrays
- Create MNE Raw objects from arrays
"""

import numpy as np
from typing import Optional, List, Union
from .loaders import SEEGData


def mat_to_mne(seeg_data: SEEGData, ch_types: Optional[Union[str, List[str]]] = 'seeg'):
    """
    Convert SEEGData to MNE-Python Raw object.
    
    Parameters
    ----------
    seeg_data : SEEGData
        Data container from load_mat_file
    ch_types : str or list of str
        Channel type(s). Default is 'seeg' for all channels.
        
    Returns
    -------
    mne.io.RawArray
        MNE-Python Raw object
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required. Install with: pip install mne")
    
    if seeg_data.is_epoched:
        raise ValueError("Data is epoched. Use mat_to_mne_epochs instead.")
    
    # Create channel types list
    n_ch = seeg_data.n_channels
    if isinstance(ch_types, str):
        ch_types_list = [ch_types] * n_ch
    else:
        ch_types_list = ch_types
    
    # Create MNE info
    info = mne.create_info(
        ch_names=seeg_data.ch_names,
        sfreq=seeg_data.sfreq,
        ch_types=ch_types_list
    )
    
    # Create Raw object
    raw = mne.io.RawArray(seeg_data.data, info, verbose=False)
    
    return raw


def mat_to_mne_epochs(
    seeg_data: SEEGData,
    ch_types: Optional[Union[str, List[str]]] = 'seeg',
    events: Optional[np.ndarray] = None,
    event_id: Optional[dict] = None,
    tmin: float = 0.0
):
    """
    Convert epoched SEEGData to MNE-Python Epochs object.
    
    Parameters
    ----------
    seeg_data : SEEGData
        Epoched data container
    ch_types : str or list of str
        Channel type(s)
    events : np.ndarray, optional
        Events array (n_events, 3). Auto-generated if not provided.
    event_id : dict, optional
        Event ID dictionary
    tmin : float
        Start time of epochs in seconds
        
    Returns
    -------
    mne.Epochs
        MNE-Python Epochs object
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required. Install with: pip install mne")
    
    if not seeg_data.is_epoched:
        raise ValueError("Data is not epoched. Use mat_to_mne instead.")
    
    n_epochs = seeg_data.n_epochs
    n_ch = seeg_data.n_channels
    
    # Create channel types list
    if isinstance(ch_types, str):
        ch_types_list = [ch_types] * n_ch
    else:
        ch_types_list = ch_types
    
    # Create MNE info
    info = mne.create_info(
        ch_names=seeg_data.ch_names,
        sfreq=seeg_data.sfreq,
        ch_types=ch_types_list
    )
    
    # Create events if not provided
    if events is None:
        events = np.column_stack([
            np.arange(n_epochs) * seeg_data.n_timepoints,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int)
        ])
    
    if event_id is None:
        event_id = {'epoch': 1}
    
    # Create Epochs object
    epochs = mne.EpochsArray(
        seeg_data.data,
        info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        verbose=False
    )
    
    return epochs


def mne_to_array(raw_or_epochs) -> SEEGData:
    """
    Convert MNE-Python Raw or Epochs to SEEGData.
    
    Parameters
    ----------
    raw_or_epochs : mne.io.Raw or mne.Epochs
        MNE-Python data object
        
    Returns
    -------
    SEEGData
        Data container
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required. Install with: pip install mne")
    
    if isinstance(raw_or_epochs, mne.io.BaseRaw):
        data = raw_or_epochs.get_data()
        times = raw_or_epochs.times
    elif isinstance(raw_or_epochs, mne.BaseEpochs):
        data = raw_or_epochs.get_data()
        times = raw_or_epochs.times
    else:
        raise TypeError(f"Expected MNE Raw or Epochs, got {type(raw_or_epochs)}")
    
    return SEEGData(
        data=data,
        sfreq=raw_or_epochs.info['sfreq'],
        ch_names=raw_or_epochs.ch_names,
        times=times,
        metadata={'mne_info': raw_or_epochs.info}
    )


def create_mne_raw(
    data: np.ndarray,
    sfreq: float,
    ch_names: Optional[List[str]] = None,
    ch_types: Optional[Union[str, List[str]]] = 'seeg'
):
    """
    Create MNE-Python Raw object from NumPy array.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_timepoints)
    sfreq : float
        Sampling frequency
    ch_names : list of str, optional
        Channel names
    ch_types : str or list of str
        Channel types
        
    Returns
    -------
    mne.io.RawArray
        MNE-Python Raw object
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required. Install with: pip install mne")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")
    
    n_channels = data.shape[0]
    
    # Generate channel names if not provided
    if ch_names is None:
        ch_names = [f'CH{i+1}' for i in range(n_channels)]
    
    # Create channel types list
    if isinstance(ch_types, str):
        ch_types_list = [ch_types] * n_channels
    else:
        ch_types_list = ch_types
    
    # Create info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types_list
    )
    
    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    return raw
