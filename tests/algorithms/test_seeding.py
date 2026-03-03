# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment.algorithms.seeding module."""

import numpy as np
import pytest

from autofragment.algorithms.seeding import (
    SEEDING_STRATEGIES,
    compute_seeds,
    get_strategy,
)


@pytest.fixture
def sample_centroids():
    """Generate sample 3D centroids for testing."""
    rng = np.random.RandomState(42)
    return rng.randn(16, 3) * 5


class TestSeedingStrategies:
    """Tests for individual seeding strategies."""

    @pytest.mark.parametrize("strategy", ["halfplane", "pca", "axis", "radial"])
    def test_correct_shape(self, sample_centroids, strategy):
        """Each strategy returns (n_clusters, 3) array."""
        seeds = compute_seeds(sample_centroids, 4, strategy)
        assert seeds.shape == (4, 3)

    @pytest.mark.parametrize("strategy", ["halfplane", "pca", "axis", "radial"])
    def test_distinct_seeds(self, sample_centroids, strategy):
        """Each strategy produces distinct seed positions."""
        seeds = compute_seeds(sample_centroids, 4, strategy)
        # No two seeds should be identical
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                assert not np.allclose(seeds[i], seeds[j]), (
                    f"Seeds {i} and {j} are identical for strategy '{strategy}'"
                )

    @pytest.mark.parametrize("n_clusters", [2, 3, 4, 8])
    def test_varying_n_clusters(self, sample_centroids, n_clusters):
        """Strategy works for various cluster counts."""
        seeds = compute_seeds(sample_centroids, n_clusters, "pca")
        assert seeds.shape == (n_clusters, 3)

    def test_axis_with_explicit_axis(self, sample_centroids):
        """Axis strategy works with explicit axis='z'."""
        seeds = compute_seeds(sample_centroids, 4, "axis", axis="z")
        assert seeds.shape == (4, 3)

    def test_axis_auto_detect(self, sample_centroids):
        """Axis strategy auto-detects when no axis given."""
        seeds = compute_seeds(sample_centroids, 4, "axis")
        assert seeds.shape == (4, 3)

    def test_axis_invalid_axis(self, sample_centroids):
        """Axis strategy raises on invalid axis name."""
        with pytest.raises(ValueError, match="axis must be"):
            compute_seeds(sample_centroids, 4, "axis", axis="w")


class TestRegistry:
    """Tests for strategy registry and dispatch."""

    def test_get_strategy_valid(self):
        """get_strategy returns callable for valid names."""
        for name in SEEDING_STRATEGIES:
            fn = get_strategy(name)
            assert callable(fn)

    def test_get_strategy_invalid(self):
        """get_strategy raises ValueError for unknown name."""
        with pytest.raises(ValueError, match="Unknown seeding strategy"):
            get_strategy("nonexistent")

    def test_compute_seeds_dispatch(self, sample_centroids):
        """compute_seeds correctly dispatches to strategy functions."""
        seeds = compute_seeds(sample_centroids, 4, "pca")
        assert seeds.shape == (4, 3)

    def test_all_strategies_registered(self):
        """All expected strategies are in the registry."""
        expected = {"halfplane", "pca", "axis", "radial"}
        assert set(SEEDING_STRATEGIES.keys()) == expected


class TestIntegrationWithClustering:
    """Test seeding strategies integrated with partition_labels."""

    def test_init_with_strategy_name(self, sample_centroids):
        """partition_labels works with init='pca'."""
        from autofragment.algorithms.clustering import partition_labels

        labels = partition_labels(sample_centroids, 4, method="kmeans", init="pca")
        assert len(labels) == 16
        assert len(set(labels)) == 4

    def test_init_with_dict(self, sample_centroids):
        """partition_labels works with init=dict."""
        from autofragment.algorithms.clustering import partition_labels

        labels = partition_labels(
            sample_centroids, 4, method="kmeans",
            init={"strategy": "axis", "axis": "z"},
        )
        assert len(labels) == 16
        assert len(set(labels)) == 4

    def test_init_with_ndarray(self, sample_centroids):
        """partition_labels works with init=ndarray."""
        from autofragment.algorithms.clustering import partition_labels

        seeds = compute_seeds(sample_centroids, 4, "pca")
        labels = partition_labels(sample_centroids, 4, method="kmeans", init=seeds)
        assert len(labels) == 16
        assert len(set(labels)) == 4

    def test_init_none_default(self, sample_centroids):
        """partition_labels with init=None uses k-means++ (default behavior)."""
        from autofragment.algorithms.clustering import partition_labels

        labels = partition_labels(sample_centroids, 4, method="kmeans", init=None)
        assert len(labels) == 16

    @pytest.mark.parametrize("method", ["agglomerative", "spectral", "gmm", "birch"])
    def test_init_ignored_for_non_kmeans(self, sample_centroids, method):
        """init is silently ignored for methods that don't support it."""
        from autofragment.algorithms.clustering import partition_labels

        labels = partition_labels(sample_centroids, 4, method=method, init="pca")
        assert len(labels) == 16
