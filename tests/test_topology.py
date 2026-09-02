"""
Unit tests for seeg_manifold.topology module

The headline property under test is that persistent homology separates a
point cloud with a hole (a noisy ring) from one without (a blob).

These tests require the optional `ripser` and `persim` dependencies and
are skipped when they are not installed.
"""

import numpy as np
import pytest

ripser = pytest.importorskip("ripser", reason="topology extras not installed")

from seeg_manifold.topology.persistent_homology import (
    compute_persistence_diagram, compute_persistence_landscape,
    compute_betti_curve, persistence_statistics, persistence_image,
    bottleneck_distance, wasserstein_distance,
)


@pytest.fixture(scope="module")
def ring():
    """Noisy circle: one prominent 1-dimensional feature."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    return np.c_[np.cos(theta), np.sin(theta)] + 0.05 * rng.normal(size=(200, 2))


@pytest.fixture(scope="module")
def blob():
    """Gaussian blob: no persistent loop."""
    rng = np.random.default_rng(1)
    return 0.3 * rng.normal(size=(200, 2))


@pytest.fixture(scope="module")
def ring_diagrams(ring):
    return compute_persistence_diagram(ring, max_dim=1)


@pytest.fixture(scope="module")
def blob_diagrams(blob):
    return compute_persistence_diagram(blob, max_dim=1)


class TestPersistenceDiagram:
    """Tests for diagram computation."""

    def test_returns_one_diagram_per_dimension(self, ring_diagrams):
        assert len(ring_diagrams) == 2          # H0 and H1
        for dgm in ring_diagrams:
            assert dgm.ndim == 2 and dgm.shape[1] == 2

    def test_births_precede_deaths(self, ring_diagrams):
        for dgm in ring_diagrams:
            assert np.all(dgm[:, 1] >= dgm[:, 0])

    def test_ring_has_a_persistent_loop(self, ring_diagrams, blob_diagrams):
        """The core claim: a hole shows up in H1, a blob has none."""
        ring_h1 = persistence_statistics(ring_diagrams)['H1']['max_persistence']
        blob_h1 = persistence_statistics(blob_diagrams)['H1']['max_persistence']

        assert ring_h1 > 10 * blob_h1

    def test_accepts_3d_input(self):
        data = np.random.default_rng(0).normal(size=(40, 3, 5))
        diagrams = compute_persistence_diagram(data, max_dim=1)

        assert len(diagrams) == 2


class TestPersistenceStatistics:
    """Tests for diagram summary statistics."""

    def test_expected_keys(self, ring_diagrams):
        stats = persistence_statistics(ring_diagrams)

        assert set(stats) == {'H0', 'H1'}
        for dim_stats in stats.values():
            assert 'n_features' in dim_stats
            assert 'max_persistence' in dim_stats
            assert 'total_persistence' in dim_stats

    def test_empty_diagram_is_handled(self):
        stats = persistence_statistics([np.empty((0, 2))])

        assert stats['H0']['n_features'] == 0
        assert stats['H0']['max_persistence'] == 0


class TestBettiCurve:
    """Tests for Betti curves."""

    def test_shape_and_nonnegative(self, ring_diagrams):
        values, betti = compute_betti_curve(ring_diagrams[1], resolution=50)

        assert values.shape == (50,) and betti.shape == (50,)
        assert np.all(betti >= 0)
        assert np.all(np.diff(values) > 0)

    def test_ring_reaches_betti_one(self, ring_diagrams):
        _, betti = compute_betti_curve(ring_diagrams[1], resolution=100)

        assert betti.max() >= 1

    def test_empty_diagram_gives_zero_curve(self):
        values, betti = compute_betti_curve(np.empty((0, 2)), resolution=10)

        assert np.all(betti == 0)
        assert values.shape == (10,)


class TestLandscapeAndImage:
    """Tests for vectorised diagram representations."""

    def test_landscape_shape(self, ring_diagrams):
        landscapes = compute_persistence_landscape(
            ring_diagrams[1], num_landscapes=3, resolution=50
        )

        assert landscapes.shape == (3, 50)
        assert np.all(landscapes >= 0)

    def test_landscapes_are_ordered(self, ring_diagrams):
        """Landscape k must dominate landscape k+1 pointwise."""
        landscapes = compute_persistence_landscape(ring_diagrams[1], num_landscapes=3)

        assert np.all(landscapes[0] >= landscapes[1] - 1e-12)
        assert np.all(landscapes[1] >= landscapes[2] - 1e-12)

    def test_empty_diagram_gives_zero_landscape(self):
        landscapes = compute_persistence_landscape(
            np.empty((0, 2)), num_landscapes=2, resolution=10
        )

        assert landscapes.shape == (2, 10)
        assert np.all(landscapes == 0)

    def test_persistence_image_shape(self, ring_diagrams):
        image = persistence_image(ring_diagrams[1], resolution=(20, 30))

        assert image.shape == (20, 30)
        assert np.all(image >= 0)


class TestDiagramDistances:
    """Tests for distances between diagrams."""

    def test_bottleneck_self_distance_is_zero(self, ring_diagrams):
        assert bottleneck_distance(ring_diagrams[1], ring_diagrams[1]) == pytest.approx(0.0)

    def test_wasserstein_self_distance_is_zero(self, ring_diagrams):
        """Regression: this wrapper used to pass an unsupported `p` argument."""
        assert wasserstein_distance(ring_diagrams[1], ring_diagrams[1]) == pytest.approx(0.0)

    def test_distances_separate_ring_from_blob(self, ring_diagrams, blob_diagrams):
        assert bottleneck_distance(ring_diagrams[1], blob_diagrams[1]) > 0
        assert wasserstein_distance(ring_diagrams[1], blob_diagrams[1]) > 0
