"""
Preprocessing Module

Functions for SEEG data preprocessing:
- Filtering (bandpass, notch)
- Artifact removal
- Epoching
- Referencing
"""

from .filters import (
    bandpass_filter,
    notch_filter,
    filter_data,
)

from .artifact_removal import (
    detect_artifacts,
    remove_artifacts_threshold,
    remove_artifacts_ica,
)

from .epoching import (
    create_epochs,
    epoch_data,
)

from .pipeline import (
    preprocess_pipeline,
    PreprocessingConfig,
)

__all__ = [
    'bandpass_filter',
    'notch_filter', 
    'filter_data',
    'detect_artifacts',
    'remove_artifacts_threshold',
    'remove_artifacts_ica',
    'create_epochs',
    'epoch_data',
    'preprocess_pipeline',
    'PreprocessingConfig',
]
