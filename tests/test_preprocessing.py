"""
Unit tests for seeg_manifold.preprocessing module

Tests for:
- Filtering functions
- Artifact detection and removal
- Epoching
- Preprocessing pipeline
"""

import pytest
import numpy as np
from scipy import signal

from seeg_manifold.preprocessing.filters import (
    bandpass_filter, highpass_filter, lowpass_filter,
    notch_filter, filter_data
)
from seeg_manifold.preprocessing.artifact_removal import (
    detect_artifacts, remove_artifacts_threshold, remove_artifacts_ica
)
from seeg_manifold.preprocessing.epoching import (
    create_epochs, epoch_data, concatenate_epochs
)
from seeg_manifold.preprocessing.pipeline import (
    preprocess_pipeline, PreprocessingConfig
)


class TestFilters:
    """Tests for filtering functions."""
    
    def test_bandpass_filter_shape(self):
        """Test that bandpass filter preserves shape."""
        data = np.random.randn(8, 1000)
        sfreq = 500.0
        
        filtered = bandpass_filter(data, sfreq, lowcut=1, highcut=100)
        
        assert filtered.shape == data.shape
    
    def test_bandpass_filter_frequency_response(self):
        """Test that bandpass filter attenuates correct frequencies."""
        sfreq = 1000.0
        n_samples = 10000
        t = np.arange(n_samples) / sfreq
        
        # Create signal with components at 5 Hz (pass), 50 Hz (pass), 200 Hz (stop)
        data = (np.sin(2 * np.pi * 5 * t) + 
                np.sin(2 * np.pi * 50 * t) + 
                np.sin(2 * np.pi * 200 * t))
        data = data.reshape(1, -1)
        
        filtered = bandpass_filter(data, sfreq, lowcut=1, highcut=100)
        
        # Compute power spectrum
        freqs, psd = signal.welch(filtered[0], sfreq, nperseg=1024)
        
        # Check that 200 Hz is attenuated
        idx_5 = np.argmin(np.abs(freqs - 5))
        idx_50 = np.argmin(np.abs(freqs - 50))
        idx_200 = np.argmin(np.abs(freqs - 200))
        
        # 200 Hz should be much smaller than 50 Hz
        assert psd[idx_200] < psd[idx_50] * 0.1
    
    def test_highpass_filter(self):
        """Test highpass filter."""
        sfreq = 500.0
        n_samples = 5000
        t = np.arange(n_samples) / sfreq
        
        # Signal with DC offset + 10 Hz oscillation
        data = 5.0 + np.sin(2 * np.pi * 10 * t)
        data = data.reshape(1, -1)
        
        filtered = highpass_filter(data, sfreq, cutoff=1)
        
        # DC offset should be removed
        assert np.abs(np.mean(filtered)) < 0.5
    
    def test_lowpass_filter(self):
        """Test lowpass filter."""
        sfreq = 1000.0
        n_samples = 5000
        t = np.arange(n_samples) / sfreq
        
        # Signal with 10 Hz + 200 Hz
        data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)
        data = data.reshape(1, -1)
        
        filtered = lowpass_filter(data, sfreq, cutoff=50)
        
        # Compute power spectrum
        freqs, psd = signal.welch(filtered[0], sfreq, nperseg=1024)
        
        # 200 Hz should be attenuated
        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_200 = np.argmin(np.abs(freqs - 200))
        
        assert psd[idx_200] < psd[idx_10] * 0.01
    
    def test_notch_filter(self):
        """Test notch filter removes line noise."""
        sfreq = 1000.0
        n_samples = 10000
        t = np.arange(n_samples) / sfreq
        
        # Signal with 10 Hz + 50 Hz line noise
        data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
        data = data.reshape(1, -1)
        
        filtered = notch_filter(data, sfreq, freq=50, harmonics=False)
        
        # Compute power spectrum
        freqs, psd = signal.welch(filtered[0], sfreq, nperseg=1024)
        
        # 50 Hz should be attenuated
        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_50 = np.argmin(np.abs(freqs - 50))
        
        assert psd[idx_50] < psd[idx_10] * 0.1
    
    def test_filter_data_combined(self):
        """Test combined filter_data function."""
        data = np.random.randn(4, 2000)
        sfreq = 500.0
        
        filtered = filter_data(data, sfreq, lowcut=1, highcut=100, notch_freq=50)
        
        assert filtered.shape == data.shape
        assert not np.array_equal(filtered, data)


class TestArtifactRemoval:
    """Tests for artifact detection and removal."""
    
    def test_detect_artifacts_continuous(self):
        """Test artifact detection in continuous data."""
        # Create clean data with one artifact
        data = np.random.randn(4, 1000) * 0.1
        data[0, 500:510] = 10.0  # Inject artifact
        
        mask, info = detect_artifacts(data, sfreq=500, threshold_std=3.0)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert np.any(mask[500:510])  # Artifact should be detected
    
    def test_detect_artifacts_epoched(self):
        """Test artifact detection in epoched data."""
        np.random.seed(42)
        # Create clean epochs with one bad epoch
        # Use very small noise so the artifact stands out more
        data = np.random.randn(20, 4, 100) * 0.01
        # Make the bad epoch have very large values that exceed any threshold
        data[5, :, :] = 100.0  # Bad epoch - much larger than normal
        
        mask, info = detect_artifacts(data, sfreq=500, threshold_std=3.0)
        
        assert mask.shape == (20,)
        # The bad epoch should definitely be detected given the large difference
        assert info['n_bad_epochs'] >= 1
        # Check that at least one bad epoch was found
        assert np.any(mask)
    
    def test_remove_artifacts_threshold_interpolate(self):
        """Test artifact removal with interpolation."""
        # Create data with artifact
        data = np.sin(np.linspace(0, 10 * np.pi, 1000)).reshape(1, -1)
        data[0, 400:410] = 100.0  # Artifact
        
        cleaned = remove_artifacts_threshold(
            data, sfreq=500, threshold_std=3.0, interpolate=True
        )
        
        # Artifact region should be interpolated
        assert np.max(np.abs(cleaned[0, 400:410])) < 10
    
    def test_remove_artifacts_threshold_epoched(self):
        """Test artifact removal for epoched data removes bad epochs."""
        data = np.random.randn(10, 4, 100) * 0.1
        data[3, :, :] = 50.0  # Bad epoch
        
        cleaned = remove_artifacts_threshold(data, sfreq=500, threshold_std=3.0)
        
        assert cleaned.shape[0] < 10  # Bad epoch removed
    
    def test_remove_artifacts_ica(self):
        """Test ICA artifact removal."""
        np.random.seed(42)
        
        # Create signal with artifact component
        n_channels, n_times = 8, 2000
        
        # Clean sources
        sources = np.random.randn(5, n_times) * 0.1
        
        # Add artifact source with high kurtosis
        artifact = np.zeros(n_times)
        artifact[500:520] = 10
        artifact[1000:1020] = 10
        sources = np.vstack([sources, artifact.reshape(1, -1)])
        
        # Mix sources
        mixing = np.random.randn(n_channels, 6)
        data = mixing @ sources
        
        cleaned, info = remove_artifacts_ica(
            data, sfreq=500, n_components=6, threshold=3.0
        )
        
        assert cleaned.shape == data.shape
        assert 'excluded_components' in info


class TestEpoching:
    """Tests for epoching functions."""
    
    def test_create_epochs_shape(self):
        """Test that epoch creation produces correct shape."""
        data = np.random.randn(8, 10000)
        sfreq = 1000.0
        
        epochs, times = create_epochs(data, sfreq, epoch_length=1.0, overlap=0.0)
        
        assert epochs.shape == (10, 8, 1000)  # 10 epochs of 1 second
        assert len(times) == 1000
    
    def test_create_epochs_overlap(self):
        """Test epoching with overlap."""
        data = np.random.randn(4, 5000)
        sfreq = 1000.0
        
        epochs_no_overlap, _ = create_epochs(data, sfreq, epoch_length=1.0, overlap=0.0)
        epochs_with_overlap, _ = create_epochs(data, sfreq, epoch_length=1.0, overlap=0.5)
        
        # More epochs with overlap
        assert epochs_with_overlap.shape[0] > epochs_no_overlap.shape[0]
    
    def test_create_epochs_event_locked(self):
        """Test event-locked epoching."""
        data = np.random.randn(4, 10000)
        sfreq = 1000.0
        
        # Events at 1s, 3s, 5s, 7s
        events = np.array([1.0, 3.0, 5.0, 7.0])
        
        epochs, times = create_epochs(
            data, sfreq, epoch_length=1.0,
            events=events, tmin=-0.2, tmax=0.8
        )
        
        assert epochs.shape[0] == 4  # 4 events
        assert epochs.shape[2] == 1000  # 1 second epochs
        
        # Time vector should span -0.2 to 0.8
        assert times[0] < 0
        assert times[-1] > 0
    
    def test_event_locked_without_tmax_raises_clearly(self):
        """Regression: this used to fail with a TypeError on None * float."""
        data = np.random.randn(4, 5000)

        with pytest.raises(ValueError, match="tmax"):
            create_epochs(data, 500.0, epoch_length=2.0,
                          events=np.array([1000, 2000]), tmin=-0.2)

    def test_epoch_data_with_rejection(self):
        """Test epoching with artifact rejection."""
        data = np.random.randn(4, 10000) * 0.1
        # Inject artifact in middle of data
        data[:, 4000:4100] = 10.0
        
        epochs, times, info = epoch_data(
            data, sfreq=1000.0, epoch_length=1.0,
            overlap=0.0, reject_threshold=3.0
        )
        
        assert info['n_epochs_rejected'] > 0
        assert info['n_epochs_final'] < info['n_epochs_original']

    def test_clean_data_survives_rejection(self):
        """Regression: rejection must not discard every epoch.

        The threshold was once ``reject_threshold * std(epoch_max)`` with
        no mean offset, which compares an absolute amplitude against a
        spread and rejected all epochs whenever their peak amplitudes
        were tightly clustered -- the normal case for clean data.
        """
        rng = np.random.default_rng(0)
        sfreq = 500.0
        t = np.arange(10000) / sfreq
        data = np.array([np.sin(2 * np.pi * 10 * t) for _ in range(4)])
        data += 0.05 * rng.standard_normal(data.shape)

        epochs, _, info = epoch_data(
            data, sfreq=sfreq, epoch_length=2.0,
            overlap=0.5, reject_threshold=5.0
        )

        assert info['n_epochs_final'] > 0, "all epochs rejected on clean data"
        assert info['n_epochs_rejected'] == 0
        assert epochs.shape[0] == info['n_epochs_original']

    def test_rejection_still_removes_outlier_epochs(self):
        """A single large artifact must still be rejected."""
        rng = np.random.default_rng(0)
        data = 0.1 * rng.standard_normal((4, 10000))
        data[:, 4000:4100] = 50.0

        _, _, info = epoch_data(
            data, sfreq=1000.0, epoch_length=1.0,
            overlap=0.0, reject_threshold=3.0
        )

        assert info['n_epochs_rejected'] > 0
        assert info['n_epochs_final'] > 0

    def test_concatenate_epochs(self):
        """Test epoch concatenation."""
        epochs = np.random.randn(5, 4, 100)
        
        continuous = concatenate_epochs(epochs)
        
        assert continuous.shape == (4, 500)
    
    def test_epoch_times_fixed_length(self):
        """Test that fixed-length epoch times are correct."""
        data = np.random.randn(4, 5000)
        sfreq = 500.0
        
        epochs, times = create_epochs(data, sfreq, epoch_length=2.0)
        
        assert times[0] == 0.0
        np.testing.assert_almost_equal(times[-1], (len(times) - 1) / sfreq)


class TestPreprocessingPipeline:
    """Tests for the preprocessing pipeline."""
    
    def test_pipeline_basic(self):
        """Test basic pipeline execution."""
        np.random.seed(42)
        data = np.random.randn(8, 5000)
        sfreq = 500.0
        
        processed, info = preprocess_pipeline(
            data, sfreq,
            lowcut=1, highcut=100,
            notch_freq=50,
            epoch_length=1.0,
            verbose=False
        )
        
        assert processed.ndim == 3  # Epoched
        assert 'filtering' in info['steps']
        assert info['sfreq'] == sfreq
    
    def test_pipeline_no_epochs(self):
        """Test pipeline without epoching."""
        data = np.random.randn(4, 2000)
        sfreq = 500.0
        
        processed, info = preprocess_pipeline(
            data, sfreq,
            lowcut=1, highcut=100,
            return_epochs=False,
            verbose=False
        )
        
        assert processed.ndim == 2  # Continuous
    
    def test_preprocessing_config(self):
        """Test PreprocessingConfig."""
        config = PreprocessingConfig(
            lowcut=0.5,
            highcut=200.0,
            notch_freq=60.0
        )
        
        assert config.lowcut == 0.5
        assert config.highcut == 200.0
        assert config.notch_freq == 60.0
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = PreprocessingConfig()
        config_dict = config.to_dict()
        
        assert 'lowcut' in config_dict
        assert 'highcut' in config_dict
        assert 'notch_freq' in config_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

