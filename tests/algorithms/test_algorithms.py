# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment.algorithms module."""

import numpy as np
import pytest

from autofragment.algorithms.clustering import partition_labels
from autofragment.algorithms.geometric import partition_by_planes
from autofragment.algorithms.hungarian import hungarian_assignment


class TestClustering:
    """Tests for clustering algorithms."""

    @pytest.fixture
    def sample_centroids(self):
        """Generate sample centroids for testing."""
        np.random.seed(42)
        return np.random.randn(16, 3)

    def test_kmeans(self, sample_centroids):
        """Test k-means clustering."""
        labels = partition_labels(sample_centroids, 4, method="kmeans")
        assert len(labels) == 16
        assert set(labels) == {0, 1, 2, 3}

    def test_agglomerative(self, sample_centroids):
        """Test agglomerative clustering."""
        labels = partition_labels(sample_centroids, 4, method="agglomerative")
        assert len(labels) == 16
        assert len(set(labels)) == 4

    def test_spectral(self, sample_centroids):
        """Test spectral clustering."""
        labels = partition_labels(sample_centroids, 4, method="spectral")
        assert len(labels) == 16
        assert len(set(labels)) == 4

    def test_gmm(self, sample_centroids):
        """Test GMM clustering."""
        labels = partition_labels(sample_centroids, 4, method="gmm")
        assert len(labels) == 16

    def test_birch(self, sample_centroids):
        """Test BIRCH clustering."""
        labels = partition_labels(sample_centroids, 4, method="birch")
        assert len(labels) == 16

    @pytest.mark.parametrize("method", ["kmeans", "agglomerative", "spectral"])
    def test_reproducibility(self, sample_centroids, method):
        """Test that clustering is reproducible with same random state."""
        labels1 = partition_labels(sample_centroids, 4, method=method, random_state=42)
        labels2 = partition_labels(sample_centroids, 4, method=method, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)

    def test_invalid_method(self, sample_centroids):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            partition_labels(sample_centroids, 4, method="invalid")

    def test_zero_groups(self, sample_centroids):
        """Test that zero groups raises error."""
        with pytest.raises(ValueError, match="n_groups must be positive"):
            partition_labels(sample_centroids, 0)

    def test_too_many_groups(self, sample_centroids):
        """Test that too many groups raises error."""
        with pytest.raises(ValueError, match="Cannot create"):
            partition_labels(sample_centroids, 100)


class TestHungarian:
    """Tests for Hungarian algorithm."""

    def test_simple_assignment(self):
        """Test simple assignment problem."""
        cost = np.array([
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ], dtype=float)
        rows, cols = hungarian_assignment(cost)
        # Optimal is diagonal: (0,0), (1,1), (2,2) -> total cost 3
        assert len(rows) == 3
        assert len(cols) == 3
        total_cost = sum(cost[r, c] for r, c in zip(rows, cols))
        assert total_cost == 3

    def test_larger_matrix(self):
        """Test larger assignment matrix."""
        np.random.seed(42)
        n = 10
        cost = np.random.rand(n, n)
        rows, cols = hungarian_assignment(cost)
        assert len(rows) == n
        assert len(cols) == n
        # Verify it's a valid assignment
        assert len(set(cols)) == n

    def test_identity_cost(self):
        """Test with identity cost matrix."""
        n = 5
        cost = np.eye(n) * 100 + 1  # High diagonal, low off-diagonal
        rows, cols = hungarian_assignment(cost)
        # Should avoid diagonal
        for r, c in zip(rows, cols):
            if r < n and c < n:
                assert cost[r, c] <= 100


class TestGeometric:
    """Tests for geometric plane partitioning."""

    @pytest.fixture
    def cube_coords(self):
        """Generate coordinates in a cube pattern."""
        # 8 points at corners of a cube
        coords = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    coords.append([x, y, z])
        return np.array(coords, dtype=float)

    def test_partition_labels(self, cube_coords):
        """Test angular slice partitioning."""
        labels = partition_by_planes(cube_coords, n_groups=4)
        assert len(labels) == 8
        assert len(set(labels)) == 4

    def test_invalid_groups(self, cube_coords):
        """Test invalid n_groups raises error."""
        with pytest.raises(ValueError, match="n_groups must be positive"):
            partition_by_planes(cube_coords, n_groups=0)
