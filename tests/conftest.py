"""
Pytest configuration and shared fixtures for SEEG Manifold tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def continuous_data():
    """Generate continuous SEEG-like data."""
    np.random.seed(42)
    n_channels = 16
    n_timepoints = 5000
    sfreq = 500.0
    
    # Create data with oscillatory components
    t = np.arange(n_timepoints) / sfreq
    data = np.zeros((n_channels, n_timepoints))
    
    for ch in range(n_channels):
        # Add alpha oscillation (8-12 Hz)
        data[ch] += np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        # Add theta oscillation (4-8 Hz)
        data[ch] += 0.5 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
        # Add noise
        data[ch] += 0.2 * np.random.randn(n_timepoints)
    
    return data, sfreq


@pytest.fixture
def epoched_data():
    """Generate epoched SEEG-like data."""
    np.random.seed(42)
    n_epochs = 30
    n_channels = 8
    n_timepoints = 500
    sfreq = 500.0
    
    data = np.random.randn(n_epochs, n_channels, n_timepoints)
    
    return data, sfreq


@pytest.fixture
def low_dim_data():
    """Generate data with known low-dimensional structure."""
    np.random.seed(42)
    n_samples = 200
    true_dim = 5
    ambient_dim = 50
    
    # Generate low-dimensional latent space
    latent = np.random.randn(n_samples, true_dim)
    
    # Random projection to high-dimensional space
    projection = np.random.randn(true_dim, ambient_dim)
    data = latent @ projection
    
    # Add small noise
    data += np.random.randn(n_samples, ambient_dim) * 0.1
    
    return data, true_dim


@pytest.fixture
def seeg_data_object():
    """Create a SEEGData object for testing."""
    from src.io.loaders import SEEGData
    
    np.random.seed(42)
    data = np.random.randn(16, 2000)
    sfreq = 500.0
    ch_names = [f'sEEG{i+1}' for i in range(16)]
    
    return SEEGData(data=data, sfreq=sfreq, ch_names=ch_names)


@pytest.fixture
def tmp_mat_file(tmp_path):
    """Create a temporary MAT file for testing."""
    import scipy.io as sio
    
    np.random.seed(42)
    data = np.random.randn(8, 1000).astype(np.float64)
    sfreq = 500.0
    
    mat_path = tmp_path / "test_data.mat"
    sio.savemat(str(mat_path), {
        'data': data,
        'sfreq': sfreq
    })
    
    return mat_path

