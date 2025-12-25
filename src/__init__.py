"""
SEEG Manifold & Symmetry Analysis Toolkit

A Python package for analyzing SEEG/iEEG data with focus on:
- Manifold learning and dimensionality reduction
- Representational similarity analysis (RSA)
- Topological data analysis (TDA)
- Symmetry and group structure detection

Example usage:
    >>> from src.io import load_seeg_data
    >>> from src.preprocessing import preprocess_pipeline
    >>> from src.manifold import compare_reductions
    
    >>> data = load_seeg_data('your_data.mat')
    >>> processed = preprocess_pipeline(data)
    >>> embeddings = compare_reductions(processed, methods=['pca', 'umap'])
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Convenient imports
from . import io
from . import preprocessing
from . import features
from . import manifold
from . import rsa
from . import topology
from . import symmetry
from . import visualization
