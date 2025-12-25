"""
Unit tests for src.manifold module

Tests for:
- Dimensionality estimation
- Reduction methods
- Multi-method comparison
"""

import pytest
import numpy as np
from scipy.spatial.distance import pdist

from src.manifold.dimensionality import (
    estimate_dimensionality, pca_explained_variance, pca_elbow,
    intrinsic_dim_mle, intrinsic_dim_correlation
)
from src.manifold.reduction import (
    reduce_dimensions, fit_pca, fit_tsne, fit_isomap
)
from src.manifold.comparison import (
    compare_reductions, compute_preservation_metrics,
    compute_neighborhood_preservation, ReductionResult
)


class TestDimensionalityEstimation:
    """Tests for dimensionality estimation."""
    
    def test_pca_explained_variance(self):
        """Test PCA variance-based dimensionality estimation."""
        np.random.seed(42)
        
        # Create data with known dimensionality
        n_samples = 200
        true_dim = 5
        
        # Generate low-dimensional latent space
        latent = np.random.randn(n_samples, true_dim)
        
        # Project to high-dimensional space with noise
        projection = np.random.randn(true_dim, 50)
        data = latent @ projection + np.random.randn(n_samples, 50) * 0.1
        
        dim, var_ratio = pca_explained_variance(data, threshold=0.95)
        
        # Estimated dimension should be close to true dimension
        assert 3 <= dim <= 10
        assert len(var_ratio) > 0
    
    def test_pca_elbow(self):
        """Test PCA elbow method."""
        np.random.seed(42)
        
        # Create data with known structure
        n_samples = 200
        true_dim = 3
        
        latent = np.random.randn(n_samples, true_dim)
        projection = np.random.randn(true_dim, 30)
        data = latent @ projection
        
        dim, eigenvalues = pca_elbow(data)
        
        assert dim > 0
        assert len(eigenvalues) > 0
        # Eigenvalues should be sorted descending
        assert eigenvalues[0] >= eigenvalues[-1]
    
    def test_intrinsic_dim_mle(self):
        """Test MLE dimensionality estimation."""
        np.random.seed(42)
        
        # Create data on a known manifold (3D sphere embedded in 10D)
        n_samples = 300
        
        # Generate points on 2-sphere (intrinsic dim = 2)
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        phi = np.random.uniform(0, np.pi, n_samples)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        sphere_3d = np.column_stack([x, y, z])
        
        # Embed in higher dimension
        embedding = np.random.randn(3, 10)
        data = sphere_3d @ embedding
        
        dim = intrinsic_dim_mle(data, k=10)
        
        # Should estimate close to 2
        assert 1.0 <= dim <= 4.0
    
    def test_estimate_dimensionality_combined(self):
        """Test combined dimensionality estimation."""
        np.random.seed(42)
        
        n_samples = 200
        true_dim = 4
        
        latent = np.random.randn(n_samples, true_dim)
        projection = np.random.randn(true_dim, 30)
        data = latent @ projection + np.random.randn(n_samples, 30) * 0.1
        
        results = estimate_dimensionality(
            data,
            methods=['pca_variance', 'pca_elbow', 'mle'],
            verbose=False
        )
        
        assert 'pca_variance' in results
        assert 'pca_elbow' in results
        assert 'mle' in results
        assert 'consensus' in results
    
    def test_estimate_dimensionality_3d_input(self):
        """Test that 3D input is handled correctly."""
        np.random.seed(42)
        
        # Epoched data shape
        data = np.random.randn(50, 8, 100)
        
        results = estimate_dimensionality(data, methods=['pca_variance'], verbose=False)
        
        assert 'pca_variance' in results


class TestReduction:
    """Tests for dimensionality reduction methods."""
    
    def test_fit_pca_shape(self):
        """Test PCA output shape."""
        data = np.random.randn(100, 50)
        
        embedding = fit_pca(data, n_components=3)
        
        assert embedding.shape == (100, 3)
    
    def test_fit_pca_return_model(self):
        """Test PCA with model return."""
        data = np.random.randn(100, 50)
        
        embedding, model = fit_pca(data, n_components=3, return_model=True)
        
        assert embedding.shape == (100, 3)
        assert hasattr(model, 'explained_variance_ratio_')
    
    def test_fit_tsne_shape(self):
        """Test t-SNE output shape."""
        np.random.seed(42)
        data = np.random.randn(50, 20)
        
        embedding = fit_tsne(data, n_components=2, perplexity=10, n_iter=250)
        
        assert embedding.shape == (50, 2)
    
    def test_fit_isomap_shape(self):
        """Test Isomap output shape."""
        data = np.random.randn(100, 30)
        
        embedding = fit_isomap(data, n_components=3, n_neighbors=10)
        
        assert embedding.shape == (100, 3)
    
    def test_reduce_dimensions_pca(self):
        """Test unified reduce_dimensions with PCA."""
        data = np.random.randn(100, 50)
        
        embedding = reduce_dimensions(data, method='pca', n_components=5)
        
        assert embedding.shape == (100, 5)
    
    def test_reduce_dimensions_3d_input(self):
        """Test reduce_dimensions with 3D input."""
        data = np.random.randn(30, 8, 50)  # Epoched data
        
        embedding = reduce_dimensions(data, method='pca', n_components=3)
        
        assert embedding.shape == (30, 3)
    
    def test_reduce_dimensions_samples_axis(self):
        """Test samples_axis parameter."""
        # Data with features on axis 0
        data = np.random.randn(50, 100)  # 50 features, 100 samples
        
        embedding = reduce_dimensions(
            data, method='pca', n_components=3, samples_axis=1
        )
        
        assert embedding.shape == (100, 3)
    
    def test_reduce_dimensions_invalid_method(self):
        """Test error on invalid method."""
        data = np.random.randn(100, 50)
        
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_dimensions(data, method='invalid_method')


class TestComparison:
    """Tests for multi-method comparison."""
    
    def test_compare_reductions_basic(self):
        """Test basic comparison functionality."""
        np.random.seed(42)
        data = np.random.randn(50, 20)
        
        results = compare_reductions(
            data,
            methods=['pca'],
            n_components=3,
            compute_metrics=True,
            verbose=False
        )
        
        assert 'pca' in results
        assert isinstance(results['pca'], ReductionResult)
        assert results['pca'].embedding.shape == (50, 3)
    
    def test_compare_reductions_multiple_methods(self):
        """Test comparison with multiple methods."""
        np.random.seed(42)
        data = np.random.randn(50, 20)
        
        results = compare_reductions(
            data,
            methods=['pca', 'isomap'],
            n_components=2,
            compute_metrics=True,
            verbose=False
        )
        
        assert 'pca' in results
        assert 'isomap' in results
    
    def test_compute_preservation_metrics(self):
        """Test distance preservation metrics."""
        np.random.seed(42)
        
        # Create original and embedded data
        original = np.random.randn(50, 20)
        embedded = fit_pca(original, n_components=3)
        
        orig_dist = pdist(original)
        embed_dist = pdist(embedded)
        
        metrics = compute_preservation_metrics(orig_dist, embed_dist)
        
        assert 'distance_correlation' in metrics
        assert 'distance_correlation_pval' in metrics
        assert -1 <= metrics['distance_correlation'] <= 1
    
    def test_compute_neighborhood_preservation(self):
        """Test neighborhood preservation metrics."""
        np.random.seed(42)
        
        # Create data where PCA should preserve neighborhoods well
        original = np.random.randn(100, 20)
        embedded = fit_pca(original, n_components=5)
        
        metrics = compute_neighborhood_preservation(original, embedded, k=5)
        
        assert 'trustworthiness' in metrics
        assert 'continuity' in metrics
        assert 0 <= metrics['trustworthiness'] <= 1
        assert 0 <= metrics['continuity'] <= 1
    
    def test_neighborhood_preservation_perfect(self):
        """Test that identical embeddings have perfect preservation."""
        np.random.seed(42)
        
        data = np.random.randn(50, 10)
        
        # Use same data as "embedding"
        metrics = compute_neighborhood_preservation(data, data, k=5)
        
        # Should be perfect or near-perfect
        assert metrics['trustworthiness'] > 0.99
        assert metrics['continuity'] > 0.99
    
    def test_reduction_result_dataclass(self):
        """Test ReductionResult dataclass."""
        embedding = np.random.randn(50, 3)
        
        result = ReductionResult(
            method='test',
            embedding=embedding,
            params={'n_components': 3},
            metrics={'test_metric': 0.5}
        )
        
        assert result.method == 'test'
        assert result.embedding.shape == (50, 3)
        assert result.params['n_components'] == 3
        assert result.metrics['test_metric'] == 0.5
    
    def test_compare_reductions_3d_input(self):
        """Test comparison with 3D epoched input."""
        np.random.seed(42)
        data = np.random.randn(30, 8, 50)  # Epoched data
        
        results = compare_reductions(
            data,
            methods=['pca'],
            n_components=3,
            verbose=False
        )
        
        assert results['pca'].embedding.shape == (30, 3)


class TestUMAP:
    """Tests for UMAP (requires umap-learn)."""
    
    @pytest.fixture
    def umap_available(self):
        """Check if UMAP is available."""
        pytest.importorskip("umap")
    
    def test_fit_umap_shape(self, umap_available):
        """Test UMAP output shape."""
        from src.manifold.reduction import fit_umap
        
        np.random.seed(42)
        data = np.random.randn(100, 30)
        
        embedding = fit_umap(data, n_components=2, n_neighbors=10)
        
        assert embedding.shape == (100, 2)
    
    def test_reduce_dimensions_umap(self, umap_available):
        """Test reduce_dimensions with UMAP."""
        np.random.seed(42)
        data = np.random.randn(100, 30)
        
        embedding = reduce_dimensions(
            data, method='umap', n_components=2
        )
        
        assert embedding.shape == (100, 2)


class TestPHATE:
    """Tests for PHATE (requires phate)."""
    
    @pytest.fixture
    def phate_available(self):
        """Check if PHATE is available."""
        pytest.importorskip("phate")
    
    def test_fit_phate_shape(self, phate_available):
        """Test PHATE output shape."""
        from src.manifold.reduction import fit_phate
        
        np.random.seed(42)
        data = np.random.randn(100, 30)
        
        embedding = fit_phate(data, n_components=2, knn=5)
        
        assert embedding.shape == (100, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

