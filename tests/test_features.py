"""
Unit tests for seeg_manifold.features module

Tests for:
- Power spectral density
- Band power (including the NumPy 2.x integration path)
- Spectral entropy and peak frequency
- Connectivity measures
"""

import numpy as np
import pytest

from seeg_manifold.features.spectral import (
    compute_psd, compute_band_power, compute_spectral_entropy,
    compute_peak_frequency, extract_spectral_features, DEFAULT_BANDS,
)
from seeg_manifold.features.connectivity import (
    compute_correlation_matrix, compute_coherence,
    compute_phase_locking_value, compute_mutual_information,
)


def _oscillatory_data(n_channels=6, n_timepoints=4000, sfreq=500.0, freq=10.0):
    """Channels sharing a dominant oscillation plus independent noise."""
    rng = np.random.default_rng(0)
    t = np.arange(n_timepoints) / sfreq
    data = np.array([np.sin(2 * np.pi * freq * t) for _ in range(n_channels)])
    data += 0.1 * rng.standard_normal((n_channels, n_timepoints))
    return data, sfreq


class TestSpectral:
    """Tests for spectral feature extraction."""

    def test_compute_psd_shape(self):
        data, sfreq = _oscillatory_data()
        freqs, psd = compute_psd(data, sfreq)

        assert psd.shape[0] == data.shape[0]
        assert freqs.shape[0] == psd.shape[-1]
        assert np.all(freqs >= 0)

    def test_compute_band_power_runs(self):
        """Band power must work on NumPy 2.x, where np.trapz was removed."""
        data, sfreq = _oscillatory_data()
        power = compute_band_power(data, sfreq)

        assert set(power) == set(DEFAULT_BANDS)
        for band, values in power.items():
            assert values.shape == (data.shape[0],)
            assert np.all(np.isfinite(values)), f"non-finite power in {band}"

    def test_band_power_normalized_sums_to_at_most_one(self):
        data, sfreq = _oscillatory_data()
        power = compute_band_power(data, sfreq, normalize=True)

        total = sum(power.values())
        # Bands cover part of the spectrum, so relative power is in (0, 1].
        assert np.all(total > 0)
        assert np.all(total <= 1.0 + 1e-9)

    def test_band_power_finds_dominant_band(self):
        """A 10 Hz oscillation should put most relative power in alpha."""
        data, sfreq = _oscillatory_data(freq=10.0)
        power = compute_band_power(data, sfreq, normalize=True)

        alpha = power['alpha']
        for band in ('delta', 'theta', 'beta', 'low_gamma'):
            assert np.all(alpha > power[band]), f"alpha not dominant over {band}"

    def test_spectral_entropy_range(self):
        data, sfreq = _oscillatory_data()
        entropy = compute_spectral_entropy(data, sfreq, normalize=True)

        assert entropy.shape == (data.shape[0],)
        assert np.all(entropy >= 0)
        assert np.all(entropy <= 1.0 + 1e-9)

    def test_peak_frequency_recovers_oscillation(self):
        data, sfreq = _oscillatory_data(freq=10.0)
        peak = compute_peak_frequency(data, sfreq)

        assert peak.shape == (data.shape[0],)
        assert np.allclose(peak, 10.0, atol=2.0)

    def test_extract_spectral_features_keys(self):
        data, sfreq = _oscillatory_data()
        features = extract_spectral_features(data, sfreq)

        for band in DEFAULT_BANDS:
            assert f'power_{band}' in features
        assert 'spectral_entropy' in features
        assert 'peak_frequency' in features


class TestConnectivity:
    """Tests for connectivity measures."""

    def test_correlation_matrix_properties(self):
        data, _ = _oscillatory_data()
        corr = compute_correlation_matrix(data)

        n = data.shape[0]
        assert corr.shape == (n, n)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T)

    def test_coherence_matrix_properties(self):
        data, sfreq = _oscillatory_data(n_channels=4, n_timepoints=2000)
        coh = compute_coherence(data, sfreq)

        assert coh.shape == (4, 4)
        assert np.allclose(coh, coh.T)
        assert np.all(coh >= 0) and np.all(coh <= 1.0 + 1e-9)

    def test_plv_properties(self):
        data, sfreq = _oscillatory_data(n_channels=4, n_timepoints=2000)
        plv = compute_phase_locking_value(data, sfreq, (8, 13))

        assert plv.shape == (4, 4)
        assert np.allclose(plv, plv.T)
        assert np.all(plv >= 0) and np.all(plv <= 1.0 + 1e-9)

    def test_plv_high_for_phase_locked_channels(self):
        """Identical channels are perfectly phase locked."""
        data, sfreq = _oscillatory_data(n_channels=1, n_timepoints=2000)
        duplicated = np.vstack([data, data])
        plv = compute_phase_locking_value(duplicated, sfreq, (8, 13))

        assert plv[0, 1] > 0.99

    def test_mutual_information_symmetric(self):
        data, _ = _oscillatory_data(n_channels=3, n_timepoints=1000)
        mi = compute_mutual_information(data, n_bins=8)

        assert mi.shape == (3, 3)
        assert np.allclose(mi, mi.T)
        assert np.all(mi[np.triu_indices(3, k=1)] >= 0)
