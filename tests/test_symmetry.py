"""
Unit tests for seeg_manifold.symmetry module

The central property under test is that the detectors report structure
only when structure is present: an unstructured cloud must come back
non-significant rather than being assigned a confident fold.
"""

import numpy as np
import pytest

from seeg_manifold.symmetry.detection import (
    detect_rotational_symmetry, detect_reflection_symmetry,
    detect_periodic_structure,
)
# Imported under an alias: pytest would otherwise collect the public
# function ``test_translation_invariance`` as a test case itself.
from seeg_manifold.symmetry.detection import (
    test_translation_invariance as translation_invariance,
)


def _k_fold_cloud(k=6, n=600, jitter=0.06, seed=0):
    """Points clustered at k evenly spaced angles on a unit ring."""
    rng = np.random.default_rng(seed)
    angles = rng.integers(0, k, n) * (2 * np.pi / k) + rng.normal(0, jitter, n)
    radii = 1.0 + rng.normal(0, 0.02, n)
    return np.c_[radii * np.cos(angles), radii * np.sin(angles)]


def _isotropic_cloud(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2))


class TestRotationalSymmetry:
    """Tests for k-fold rotational symmetry detection."""

    def test_detects_true_fold(self):
        result = detect_rotational_symmetry(
            _k_fold_cloud(k=6), n_permutations=300, random_state=0
        )

        assert result['best_fold'] == 6
        assert result['significant'] is True
        assert result['best_p_value'] < 0.05

    def test_noise_is_not_significant(self):
        """Regression: unstructured data must not yield a confident fold."""
        result = detect_rotational_symmetry(
            _isotropic_cloud(), n_permutations=300, random_state=0
        )

        assert result['significant'] is False
        assert result['best_p_value'] > 0.05

    def test_p_values_are_valid_probabilities(self):
        result = detect_rotational_symmetry(
            _k_fold_cloud(k=4), n_permutations=100, random_state=0
        )

        for fold, p in result['p_values'].items():
            assert 0.0 < p <= 1.0, f"fold {fold} has invalid p-value {p}"

    def test_permutations_can_be_disabled(self):
        result = detect_rotational_symmetry(
            _k_fold_cloud(), n_permutations=0
        )

        assert np.isnan(result['best_p_value'])
        assert result['significant'] is False

    def test_reproducible_with_seed(self):
        cloud = _k_fold_cloud(k=3)
        a = detect_rotational_symmetry(cloud, n_permutations=100, random_state=7)
        b = detect_rotational_symmetry(cloud, n_permutations=100, random_state=7)

        assert a['best_p_value'] == b['best_p_value']

    def test_accepts_3d_embedding(self):
        cloud = _k_fold_cloud(k=5)
        cloud_3d = np.c_[cloud, 0.01 * np.random.default_rng(0).standard_normal(len(cloud))]
        result = detect_rotational_symmetry(
            cloud_3d, n_permutations=100, random_state=0
        )

        assert result['best_fold'] in result['fold_scores']


class TestReflectionSymmetry:
    """Tests for reflection symmetry detection."""

    def test_score_is_not_trivially_one(self):
        """Regression: the score used to be normalised by its own max."""
        result = detect_reflection_symmetry(
            _k_fold_cloud(k=6), n_permutations=50, random_state=0
        )

        assert result['best_score'] != pytest.approx(1.0)
        assert result['best_score'] >= 0.0

    def test_symmetric_cloud_is_significant(self):
        result = detect_reflection_symmetry(
            _k_fold_cloud(k=6), n_permutations=100, random_state=0
        )

        assert result['significant'] is True

    def test_noise_is_not_significant(self):
        result = detect_reflection_symmetry(
            _isotropic_cloud(n=300), n_permutations=100, random_state=0
        )

        assert result['significant'] is False

    def test_requires_2d(self):
        with pytest.raises(ValueError):
            detect_reflection_symmetry(np.random.randn(50, 3), n_permutations=0)


class TestPeriodicStructure:
    """Tests for periodicity detection."""

    def test_autocorr_recovers_period(self):
        period = 50
        t = np.arange(1000)
        signal = np.sin(2 * np.pi * t / period).reshape(-1, 1)

        result = detect_periodic_structure(signal, method='autocorr')

        assert result['period'] == pytest.approx(period, abs=3)

    def test_fft_recovers_period(self):
        period = 40
        t = np.arange(1200)
        signal = np.sin(2 * np.pi * t / period).reshape(-1, 1)

        result = detect_periodic_structure(signal, method='fft')

        assert result['period'] == pytest.approx(period, abs=4)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            detect_periodic_structure(np.random.randn(100, 1), method='nope')


class TestTranslationInvariance:
    """Tests for translation invariance."""

    def test_periodic_signal_invariant_at_full_period(self):
        period = 50
        t = np.arange(1000)
        data = np.sin(2 * np.pi * t / period).reshape(1, -1)

        similarity = translation_invariance(data, shift=period)

        assert similarity > 0.99

    def test_shift_beyond_length_raises(self):
        with pytest.raises(ValueError):
            translation_invariance(np.random.randn(2, 100), shift=200)
