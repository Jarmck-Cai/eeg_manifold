"""
Data Input/Output Module

Handles loading SEEG data from various formats:
- .mat (MATLAB files) - Recommended
- .edf (European Data Format)
- .fif (MNE-Python format)
"""

from .loaders import (
    load_seeg_data,
    load_mat_file,
    load_edf_file,
    SEEGData,
)

from .converters import (
    mat_to_mne,
    mne_to_array,
    create_mne_raw,
)

__all__ = [
    'load_seeg_data',
    'load_mat_file', 
    'load_edf_file',
    'SEEGData',
    'mat_to_mne',
    'mne_to_array',
    'create_mne_raw',
]
