# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Clustering algorithms for molecular partitioning.

This module provides a unified interface to various clustering algorithms
from scikit-learn and k-means-constrained.

Supported methods:
- kmeans: Standard k-means clustering
- kmeans_constrained: K-means with balanced cluster sizes
- agglomerative: Agglomerative clustering
- spectral: Spectral clustering
- gmm: Gaussian Mixture Model
- birch: BIRCH clustering
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

CLUSTERING_METHODS: List[str] = [
    "kmeans",
    "kmeans_constrained",
    "agglomerative",
    "spectral",
    "gmm",
    "birch",
]


def _resolve_init(
    centroids: np.ndarray,
    n_groups: int,
    init: Union[None, str, np.ndarray, Dict[str, Any]],
) -> Union[str, np.ndarray]:
    """Resolve an *init* specification to a value accepted by KMeans / KMeansConstrained.

    Returns ``"k-means++"`` (the default) when *init* is ``None``, a numpy
    seed array when a strategy name or dict is given, or the array as-is if
    one was provided directly.
    """
    if init is None:
        return "k-means++"

    if isinstance(init, np.ndarray):
        return init

    if isinstance(init, str):
        from autofragment.algorithms.seeding import compute_seeds

        return compute_seeds(centroids, n_groups, init)

    if isinstance(init, dict):
        from autofragment.algorithms.seeding import compute_seeds

        spec = dict(init)  # copy so we don't mutate caller's dict
        strategy = spec.pop("strategy")
        return compute_seeds(centroids, n_groups, strategy, **spec)

    raise TypeError(
        f"init must be None, a string, an ndarray, or a dict, got {type(init).__name__}"
    )


def partition_labels(
    centroids: np.ndarray,
    n_groups: int,
    method: str = "kmeans_constrained",
    random_state: int = 42,
    init: Union[None, str, np.ndarray, Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Partition centroids into groups using the specified clustering method.

    Parameters
    ----------
    centroids : np.ndarray
        Centroid coordinates with shape (N, 3).
    n_groups : int
        Number of groups/clusters to create.
    method : str, optional
        Clustering method name. Default is "kmeans_constrained".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    init : None | str | ndarray | dict, optional
        Initialization for k-means variants.  Accepted values:

        - ``None`` (default) -- use ``"k-means++"``.
        - A strategy name (``"halfplane"``, ``"pca"``, ``"axis"``,
          ``"radial"``) -- compute seeds via the seeding module.
        - An ``(n_groups, 3)`` ndarray of initial centroids.
        - A dict ``{"strategy": "<name>", ...}`` with optional extra
          keyword arguments forwarded to the strategy function.

        Silently ignored for methods that do not support custom
        initialization (agglomerative, spectral, gmm, birch).

    Returns
    -------
    np.ndarray
        Cluster labels with shape (N,), values in range [0, n_groups-1].

    Raises
    ------
    ValueError
        If n_groups is not positive or method is unknown.
    ImportError
        If kmeans_constrained is requested but not installed.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")

    if len(centroids) < n_groups:
        raise ValueError(
            f"Cannot create {n_groups} groups from {len(centroids)} points"
        )

    if method == "kmeans":
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]

        resolved = _resolve_init(centroids, n_groups, init)
        return KMeans(
            n_clusters=n_groups, random_state=random_state, n_init="auto",
            init=resolved,
        ).fit_predict(centroids)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]

        return AgglomerativeClustering(n_clusters=n_groups).fit_predict(centroids)

    if method == "spectral":
        from sklearn.cluster import SpectralClustering  # type: ignore[import-untyped]

        return SpectralClustering(
            n_clusters=n_groups, random_state=random_state, assign_labels="discretize"
        ).fit_predict(centroids)

    if method == "kmeans_constrained":
        try:
            from k_means_constrained import KMeansConstrained  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "kmeans_constrained requires the k-means-constrained package. "
                "Install it with: pip install k-means-constrained --no-deps"
            ) from e

        min_size = len(centroids) // n_groups
        max_size = min_size + (1 if len(centroids) % n_groups else 0)
        resolved = _resolve_init(centroids, n_groups, init)
        return KMeansConstrained(
            n_clusters=n_groups,
            size_min=min_size,
            size_max=max_size,
            random_state=random_state,
            init=resolved,
        ).fit_predict(centroids)

    if method == "gmm":
        from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]

        model = GaussianMixture(n_components=n_groups, random_state=random_state)
        return model.fit_predict(centroids)

    if method == "birch":
        from sklearn.cluster import Birch  # type: ignore[import-untyped]

        return Birch(n_clusters=n_groups).fit_predict(centroids)

    raise ValueError(f"Unknown clustering method: {method}")
