"""
SEEG Manifold & Symmetry Analysis Toolkit

A Python package for analyzing SEEG/iEEG data with focus on:
- Manifold learning and dimensionality reduction
- Representational similarity analysis (RSA)
- Topological data analysis (TDA)
- Symmetry and group structure detection

Example usage:
    >>> from seeg_manifold.io import load_seeg_data
    >>> from seeg_manifold.preprocessing import preprocess_pipeline
    >>> from seeg_manifold.manifold import compare_reductions

    >>> data = load_seeg_data('your_data.mat')
    >>> processed, info = preprocess_pipeline(data.data, sfreq=data.sfreq)
    >>> results = compare_reductions(processed, methods=['pca', 'umap'])
"""

__version__ = "0.1.0"

# Convenient imports
from . import io
from . import preprocessing
from . import features
from . import manifold
from . import rsa
from . import topology
from . import symmetry
from . import visualization
