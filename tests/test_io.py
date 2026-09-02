"""
Unit tests for seeg_manifold.io module

Tests for:
- SEEGData dataclass
- Data loaders (mat, edf, fif)
- Format converters
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from seeg_manifold.io.loaders import SEEGData, load_seeg_data, load_mat_file


class TestSEEGData:
    """Tests for SEEGData dataclass."""
    
    def test_basic_creation(self):
        """Test basic SEEGData creation with minimal params."""
        data = np.random.randn(64, 1000)
        sfreq = 1000.0
        
        seeg = SEEGData(data=data, sfreq=sfreq)
        
        assert seeg.n_channels == 64
        assert seeg.n_timepoints == 1000
        assert seeg.sfreq == 1000.0
        assert seeg.duration == 1.0
        assert not seeg.is_epoched
        assert seeg.n_epochs is None
    
    def test_epoched_data(self):
        """Test SEEGData with epoched data."""
        data = np.random.randn(50, 64, 500)  # 50 epochs, 64 channels, 500 timepoints
        sfreq = 500.0
        
        seeg = SEEGData(data=data, sfreq=sfreq)
        
        assert seeg.n_epochs == 50
        assert seeg.n_channels == 64
        assert seeg.n_timepoints == 500
        assert seeg.is_epoched
        assert seeg.duration == 1.0
    
    def test_auto_channel_names(self):
        """Test automatic channel name generation."""
        data = np.random.randn(8, 100)
        seeg = SEEGData(data=data, sfreq=100.0)
        
        assert len(seeg.ch_names) == 8
        assert seeg.ch_names[0] == 'CH1'
        assert seeg.ch_names[-1] == 'CH8'
    
    def test_custom_channel_names(self):
        """Test custom channel names."""
        data = np.random.randn(3, 100)
        ch_names = ['A1', 'A2', 'A3']
        seeg = SEEGData(data=data, sfreq=100.0, ch_names=ch_names)
        
        assert seeg.ch_names == ch_names
    
    def test_auto_times(self):
        """Test automatic time vector generation."""
        data = np.random.randn(4, 200)
        sfreq = 100.0
        seeg = SEEGData(data=data, sfreq=sfreq)
        
        assert seeg.times is not None
        assert len(seeg.times) == 200
        assert seeg.times[0] == 0.0
        np.testing.assert_almost_equal(seeg.times[-1], 1.99, decimal=2)
    
    def test_get_channel_data_continuous(self):
        """Test getting data for specific channel (continuous)."""
        data = np.random.randn(4, 100)
        ch_names = ['A', 'B', 'C', 'D']
        seeg = SEEGData(data=data, sfreq=100.0, ch_names=ch_names)
        
        ch_data = seeg.get_channel_data('B')
        np.testing.assert_array_equal(ch_data, data[1])
    
    def test_get_channel_data_epoched(self):
        """Test getting data for specific channel (epoched)."""
        data = np.random.randn(10, 4, 100)
        ch_names = ['A', 'B', 'C', 'D']
        seeg = SEEGData(data=data, sfreq=100.0, ch_names=ch_names)
        
        ch_data = seeg.get_channel_data('C')
        assert ch_data.shape == (10, 100)
        np.testing.assert_array_equal(ch_data, data[:, 2, :])
    
    def test_get_channel_data_invalid(self):
        """Test error on invalid channel name."""
        data = np.random.randn(4, 100)
        ch_names = ['A', 'B', 'C', 'D']
        seeg = SEEGData(data=data, sfreq=100.0, ch_names=ch_names)
        
        with pytest.raises(ValueError, match="Channel 'X' not found"):
            seeg.get_channel_data('X')
    
    def test_repr(self):
        """Test string representation."""
        data = np.random.randn(64, 1000)
        seeg = SEEGData(data=data, sfreq=1000.0)
        
        repr_str = repr(seeg)
        assert '64 channels' in repr_str
        assert '1000 timepoints' in repr_str
        assert '1000' in repr_str  # sfreq


class TestLoadMatFile:
    """Tests for MAT file loading."""
    
    def test_load_simple_mat(self, tmp_path):
        """Test loading simple MAT file."""
        import scipy.io as sio
        
        # Create test data
        data = np.random.randn(32, 500).astype(np.float64)
        sfreq = 500.0
        
        # Save as MAT file
        mat_path = tmp_path / "test_data.mat"
        sio.savemat(str(mat_path), {
            'data': data,
            'sfreq': sfreq
        })
        
        # Load and verify
        seeg = load_seeg_data(str(mat_path))
        
        assert seeg.n_channels == 32
        assert seeg.n_timepoints == 500
        assert seeg.sfreq == 500.0
    
    def test_load_struct_mat(self, tmp_path):
        """Test loading structured MAT file."""
        import scipy.io as sio
        
        # Create test data
        data = np.random.randn(16, 1000).astype(np.float64)
        
        # Save as structured MAT file
        mat_path = tmp_path / "test_struct.mat"
        seeg_struct = {
            'data': data,
            'sfreq': 1000.0,
            'ch_names': np.array(['CH1', 'CH2'], dtype=object)  # Simplified for test
        }
        sio.savemat(str(mat_path), {'seeg_data': seeg_struct})
        
        # Load and verify
        seeg = load_seeg_data(str(mat_path))
        
        assert seeg.n_channels == 16
        assert seeg.sfreq == 1000.0
    
    def test_file_not_found(self):
        """Test error on non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_seeg_data("nonexistent_file.mat")
    
    def test_unsupported_format(self, tmp_path):
        """Test error on unsupported file format."""
        # Create a dummy file
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_seeg_data(str(bad_path))


class TestConverters:
    """Tests for data format converters."""
    
    def test_create_mne_raw(self):
        """Test creating MNE Raw from array."""
        pytest.importorskip("mne")
        from seeg_manifold.io.converters import create_mne_raw
        
        data = np.random.randn(8, 1000)
        sfreq = 500.0
        
        raw = create_mne_raw(data, sfreq)
        
        assert raw.info['sfreq'] == 500.0
        assert len(raw.ch_names) == 8
        assert raw.get_data().shape == (8, 1000)
    
    def test_mat_to_mne(self):
        """Test converting SEEGData to MNE Raw."""
        pytest.importorskip("mne")
        from seeg_manifold.io.converters import mat_to_mne
        
        data = np.random.randn(4, 500)
        seeg = SEEGData(data=data, sfreq=250.0, ch_names=['A1', 'A2', 'B1', 'B2'])
        
        raw = mat_to_mne(seeg)
        
        assert raw.info['sfreq'] == 250.0
        assert raw.ch_names == ['A1', 'A2', 'B1', 'B2']
    
    def test_mne_to_array(self):
        """Test converting MNE Raw back to SEEGData."""
        pytest.importorskip("mne")
        import mne
        from seeg_manifold.io.converters import mne_to_array
        
        # Create MNE Raw
        data = np.random.randn(4, 500)
        info = mne.create_info(['A', 'B', 'C', 'D'], 250.0, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Convert back
        seeg = mne_to_array(raw)
        
        assert seeg.sfreq == 250.0
        assert seeg.n_channels == 4
        np.testing.assert_array_almost_equal(seeg.data, data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

