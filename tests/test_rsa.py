"""
Unit tests for seeg_manifold.rsa module

Tests for representational dissimilarity matrices, model comparison and
MDS visualisation of RDMs.
"""

import numpy as np
import pytest

from seeg_manifold.rsa.rdm import (
    compute_rdm, compute_rdm_timeseries, compare_rdms,
    compare_rdm_to_models, create_model_rdm, rdm_mds,
)


def _clustered_patterns(n_cond=4, n_per=5, n_features=20, noise=0.1, seed=0):
    """Condition patterns in n_cond well-separated clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_cond, n_features))
    labels = np.repeat(np.arange(n_cond), n_per)
    data = centers[labels] + noise * rng.normal(size=(len(labels), n_features))
    return data, labels


class TestComputeRDM:
    """Tests for RDM construction."""

    @pytest.mark.parametrize("metric", ["correlation", "euclidean", "cosine"])
    def test_rdm_is_square_symmetric_zero_diagonal(self, metric):
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data, metric=metric)

        n = data.shape[0]
        assert rdm.shape == (n, n)
        assert np.allclose(rdm, rdm.T)
        assert np.allclose(np.diag(rdm), 0)

    def test_normalize_bounds_values(self):
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data, metric='euclidean', normalize=True)

        assert rdm.max() <= 1.0 + 1e-9
        assert rdm.min() >= 0.0

    def test_flattens_3d_input(self):
        data = np.random.default_rng(0).normal(size=(6, 4, 10))
        rdm = compute_rdm(data)

        assert rdm.shape == (6, 6)

    def test_within_cluster_less_than_between(self):
        """Same-condition patterns must be more similar than different ones."""
        data, labels = _clustered_patterns(noise=0.05)
        rdm = compute_rdm(data, metric='correlation')

        same = labels[:, None] == labels[None, :]
        off_diag = ~np.eye(len(labels), dtype=bool)
        within = rdm[same & off_diag].mean()
        between = rdm[~same].mean()

        assert within < between


class TestRDMTimeseries:
    """Tests for time-resolved RDMs."""

    def test_shape_and_time_indices(self):
        data = np.random.default_rng(0).normal(size=(4, 8, 50))
        rdms, times = compute_rdm_timeseries(data, window_size=10, step=5)

        n_windows = (50 - 10) // 5 + 1
        assert rdms.shape == (n_windows, 4, 4)
        assert times.shape == (n_windows,)
        assert np.all(np.diff(times) > 0)

    def test_requires_3d(self):
        with pytest.raises(ValueError):
            compute_rdm_timeseries(np.random.randn(10, 10))


class TestModelComparison:
    """Tests for comparing RDMs against models."""

    def test_categorical_model_structure(self):
        labels = np.array([0, 0, 1, 1])
        model = create_model_rdm(labels, 'categorical')

        assert model.shape == (4, 4)
        assert model[0, 1] == 0      # same category
        assert model[0, 2] == 1      # different category

    def test_ordinal_model_is_normalized(self):
        model = create_model_rdm(np.array([0, 1, 2, 3]), 'ordinal')

        assert model.max() == pytest.approx(1.0)
        assert np.allclose(np.diag(model), 0)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            create_model_rdm(np.array([0, 1]), 'nonsense')

    def test_identical_rdms_correlate_perfectly(self):
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data)

        corr, _ = compare_rdms(rdm, rdm)

        assert corr == pytest.approx(1.0)

    def test_data_rdm_matches_its_own_category_model(self):
        data, labels = _clustered_patterns(noise=0.05)
        rdm = compute_rdm(data)
        model = create_model_rdm(labels, 'categorical')

        corr, pval = compare_rdms(rdm, model, method='spearman')

        assert corr > 0.5
        assert pval < 0.01

    @pytest.mark.parametrize("method", ["spearman", "pearson", "kendall"])
    def test_comparison_methods_run(self, method):
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data)

        corr, pval = compare_rdms(rdm, rdm, method=method)

        assert corr == pytest.approx(1.0)

    def test_unknown_comparison_method_raises(self):
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data)
        with pytest.raises(ValueError):
            compare_rdms(rdm, rdm, method='nope')

    def test_compare_to_multiple_models(self):
        data, labels = _clustered_patterns()
        rdm = compute_rdm(data)
        models = {
            'category': create_model_rdm(labels, 'categorical'),
            'ordinal': create_model_rdm(labels, 'ordinal'),
        }

        results = compare_rdm_to_models(rdm, models)

        assert set(results) == {'category', 'ordinal'}
        for stats in results.values():
            assert 'correlation' in stats and 'p_value' in stats


class TestRDMMDS:
    """Tests for MDS embedding of an RDM."""

    def test_mds_shape(self):
        """Also guards the scikit-learn MDS parameter rename."""
        data, _ = _clustered_patterns()
        rdm = compute_rdm(data)

        embedding = rdm_mds(rdm, n_components=2)

        assert embedding.shape == (data.shape[0], 2)
        assert np.all(np.isfinite(embedding))

    def test_mds_preserves_cluster_structure(self):
        data, labels = _clustered_patterns(noise=0.05)
        embedding = rdm_mds(compute_rdm(data), n_components=2)

        from scipy.spatial.distance import pdist, squareform
        d = squareform(pdist(embedding))
        same = labels[:, None] == labels[None, :]
        off_diag = ~np.eye(len(labels), dtype=bool)

        assert d[same & off_diag].mean() < d[~same].mean()
