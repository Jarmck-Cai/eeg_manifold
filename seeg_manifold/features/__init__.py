"""
Feature Extraction Module

Tools for extracting features from SEEG data:
- Spectral features (band power, entropy, peak frequency)
- Connectivity features (correlation, coherence, PLV)
"""

from .spectral import (
    compute_psd,
    compute_band_power,
    compute_spectral_entropy,
    compute_peak_frequency,
    extract_spectral_features,
    DEFAULT_BANDS,
)

from .connectivity import (
    compute_correlation_matrix,
    compute_coherence,
    compute_phase_locking_value,
    compute_mutual_information,
    extract_connectivity_features,
)

__all__ = [
    # Spectral
    'compute_psd',
    'compute_band_power',
    'compute_spectral_entropy',
    'compute_peak_frequency',
    'extract_spectral_features',
    'DEFAULT_BANDS',
    # Connectivity
    'compute_correlation_matrix',
    'compute_coherence',
    'compute_phase_locking_value',
    'compute_mutual_information',
    'extract_connectivity_features',
]
