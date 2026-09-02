"""
Data Loaders for SEEG data

Supports loading from:
- MATLAB .mat files (recommended for SEEG)
- EDF files
- MNE-Python .fif files
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
import warnings


@dataclass
class SEEGData:
    """
    Container class for SEEG data.
    
    Attributes
    ----------
    data : np.ndarray
        Neural data with shape (n_channels, n_timepoints) or 
        (n_epochs, n_channels, n_timepoints)
    sfreq : float
        Sampling frequency in Hz
    ch_names : List[str]
        Channel names
    times : Optional[np.ndarray]
        Time vector in seconds
    metadata : Dict[str, Any]
        Additional metadata (subject info, recording info, etc.)
    """
    data: np.ndarray
    sfreq: float
    ch_names: List[str] = field(default_factory=list)
    times: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Generate default channel names if not provided
        if len(self.ch_names) == 0:
            n_channels = self.data.shape[-2] if self.data.ndim >= 2 else self.data.shape[0]
            self.ch_names = [f'CH{i+1}' for i in range(n_channels)]
        
        # Generate time vector if not provided
        if self.times is None:
            n_timepoints = self.data.shape[-1]
            self.times = np.arange(n_timepoints) / self.sfreq
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        if self.data.ndim == 2:
            return self.data.shape[0]
        elif self.data.ndim == 3:
            return self.data.shape[1]
        else:
            raise ValueError(f"Unexpected data dimensions: {self.data.ndim}")
    
    @property
    def n_timepoints(self) -> int:
        """Number of timepoints."""
        return self.data.shape[-1]
    
    @property
    def n_epochs(self) -> Optional[int]:
        """Number of epochs (None if continuous data)."""
        if self.data.ndim == 3:
            return self.data.shape[0]
        return None
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_timepoints / self.sfreq
    
    @property
    def is_epoched(self) -> bool:
        """Whether data is epoched."""
        return self.data.ndim == 3
    
    def get_channel_data(self, ch_name: str) -> np.ndarray:
        """Get data for a specific channel."""
        if ch_name not in self.ch_names:
            raise ValueError(f"Channel '{ch_name}' not found. Available: {self.ch_names}")
        idx = self.ch_names.index(ch_name)
        if self.is_epoched:
            return self.data[:, idx, :]
        else:
            return self.data[idx, :]
    
    def __repr__(self) -> str:
        shape_str = f"({self.n_channels} channels, {self.n_timepoints} timepoints)"
        if self.is_epoched:
            shape_str = f"({self.n_epochs} epochs, {self.n_channels} channels, {self.n_timepoints} timepoints)"
        return f"SEEGData{shape_str} @ {self.sfreq}Hz, duration={self.duration:.2f}s"


def load_seeg_data(
    filepath: Union[str, Path],
    sfreq: Optional[float] = None,
    **kwargs
) -> SEEGData:
    """
    Load SEEG data from file, automatically detecting format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to data file
    sfreq : float, optional
        Sampling frequency (required for some formats)
    **kwargs
        Additional arguments passed to format-specific loaders
        
    Returns
    -------
    SEEGData
        Loaded data container
        
    Examples
    --------
    >>> data = load_seeg_data('recording.mat')
    >>> print(data)
    SEEGData(128 channels, 100000 timepoints) @ 1000Hz
    
    >>> data = load_seeg_data('recording.edf')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.mat':
        return load_mat_file(filepath, sfreq=sfreq, **kwargs)
    elif suffix == '.edf':
        return load_edf_file(filepath, **kwargs)
    elif suffix in ['.fif', '.fif.gz']:
        return load_fif_file(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_mat_file(
    filepath: Union[str, Path],
    sfreq: Optional[float] = None,
    data_key: Optional[str] = None,
    struct_key: str = 'seeg_data',
) -> SEEGData:
    """
    Load SEEG data from MATLAB .mat file.
    
    Expected MATLAB structure:
    - seeg_data.data: (n_channels x n_timepoints) or (n_epochs x n_channels x n_timepoints)
    - seeg_data.sfreq: sampling rate
    - seeg_data.ch_names: channel names (optional)
    - seeg_data.times: time vector (optional)
    
    Or simple format:
    - data: raw data array
    - sfreq parameter must be provided
    
    Parameters
    ----------
    filepath : str or Path
        Path to .mat file
    sfreq : float, optional
        Sampling frequency (used if not in file)
    data_key : str, optional
        Key for data array if not using struct format
    struct_key : str
        Key for SEEG struct in mat file
        
    Returns
    -------
    SEEGData
        Loaded data container
    """
    import scipy.io as sio
    import h5py
    
    filepath = Path(filepath)
    
    # Try loading with scipy first (for v7.2 and earlier)
    try:
        mat_contents = sio.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        return _parse_mat_contents(mat_contents, sfreq, data_key, struct_key)
    except NotImplementedError:
        # v7.3 files need h5py
        pass
    
    # Try loading with h5py (for v7.3 files)
    try:
        with h5py.File(filepath, 'r') as f:
            return _parse_h5_contents(f, sfreq, data_key, struct_key)
    except Exception as e:
        raise IOError(f"Failed to load MAT file: {e}")


def _parse_mat_contents(
    mat_contents: dict,
    sfreq: Optional[float],
    data_key: Optional[str],
    struct_key: str
) -> SEEGData:
    """Parse contents loaded from .mat file with scipy."""
    
    # Try structured format first
    if struct_key in mat_contents:
        struct = mat_contents[struct_key]
        data = np.array(struct.data)
        file_sfreq = float(struct.sfreq) if hasattr(struct, 'sfreq') else None
        
        ch_names = []
        if hasattr(struct, 'ch_names'):
            ch_names = list(struct.ch_names) if hasattr(struct.ch_names, '__iter__') else []
        
        times = None
        if hasattr(struct, 'times'):
            times = np.array(struct.times)
        
        metadata = {}
        for attr in dir(struct):
            if not attr.startswith('_') and attr not in ['data', 'sfreq', 'ch_names', 'times']:
                try:
                    metadata[attr] = getattr(struct, attr)
                except:
                    pass
        
        final_sfreq = sfreq or file_sfreq
        if final_sfreq is None:
            raise ValueError("Sampling frequency not found in file and not provided")
        
        return SEEGData(
            data=data,
            sfreq=final_sfreq,
            ch_names=ch_names,
            times=times,
            metadata=metadata
        )
    
    # Try simple format
    if data_key:
        data = np.array(mat_contents[data_key])
    else:
        # Try common variable names
        for key in ['data', 'Data', 'EEG', 'eeg', 'seeg', 'SEEG', 'ieeg', 'iEEG']:
            if key in mat_contents:
                data = np.array(mat_contents[key])
                break
        else:
            # Use first non-system variable
            for key, val in mat_contents.items():
                if not key.startswith('__') and isinstance(val, np.ndarray):
                    data = val
                    break
            else:
                raise ValueError("Could not find data array in MAT file")
    
    if sfreq is None:
        # Try to find sfreq in file
        for key in ['sfreq', 'Fs', 'fs', 'srate', 'sampling_rate']:
            if key in mat_contents:
                sfreq = float(mat_contents[key])
                break
        else:
            raise ValueError("Sampling frequency not found in file and not provided")
    
    return SEEGData(data=data, sfreq=sfreq)


def _parse_h5_contents(
    f: 'h5py.File',
    sfreq: Optional[float],
    data_key: Optional[str],
    struct_key: str
) -> SEEGData:
    """Parse contents from HDF5/.mat v7.3 file."""
    
    # Try structured format
    if struct_key in f:
        grp = f[struct_key]
        data = np.array(grp['data']).T  # HDF5 stores transposed
        
        file_sfreq = None
        if 'sfreq' in grp:
            file_sfreq = float(np.array(grp['sfreq']).flat[0])
        
        final_sfreq = sfreq or file_sfreq
        if final_sfreq is None:
            raise ValueError("Sampling frequency not found in file and not provided")
        
        ch_names = []
        if 'ch_names' in grp:
            # Handle MATLAB cell array of strings
            ch_refs = grp['ch_names']
            try:
                for ref in ch_refs.flat:
                    ch_names.append(''.join(chr(c) for c in f[ref][:].flat))
            except:
                pass
        
        times = None
        if 'times' in grp:
            times = np.array(grp['times']).flatten()
        
        return SEEGData(
            data=data,
            sfreq=final_sfreq,
            ch_names=ch_names,
            times=times
        )
    
    # Simple format
    if data_key and data_key in f:
        data = np.array(f[data_key]).T
    else:
        for key in ['data', 'Data', 'EEG', 'eeg', 'seeg', 'SEEG']:
            if key in f:
                data = np.array(f[key]).T
                break
        else:
            raise ValueError("Could not find data array in MAT file")
    
    if sfreq is None:
        for key in ['sfreq', 'Fs', 'fs', 'srate']:
            if key in f:
                sfreq = float(np.array(f[key]).flat[0])
                break
        else:
            raise ValueError("Sampling frequency not found")
    
    return SEEGData(data=data, sfreq=sfreq)


def load_edf_file(filepath: Union[str, Path]) -> SEEGData:
    """
    Load SEEG data from EDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .edf file
        
    Returns
    -------
    SEEGData
        Loaded data container
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required to load EDF files. Install with: pip install mne")
    
    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    
    return SEEGData(
        data=raw.get_data(),
        sfreq=raw.info['sfreq'],
        ch_names=raw.ch_names,
        times=raw.times,
        metadata={'mne_info': raw.info}
    )


def load_fif_file(filepath: Union[str, Path]) -> SEEGData:
    """
    Load SEEG data from MNE-Python .fif file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .fif file
        
    Returns
    -------
    SEEGData
        Loaded data container
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required to load FIF files. Install with: pip install mne")
    
    raw = mne.io.read_raw_fif(str(filepath), preload=True, verbose=False)
    
    return SEEGData(
        data=raw.get_data(),
        sfreq=raw.info['sfreq'],
        ch_names=raw.ch_names,
        times=raw.times,
        metadata={'mne_info': raw.info}
    )
