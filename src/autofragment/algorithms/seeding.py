# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Seeding strategies for k-means initialization.

Provides pluggable strategies that generate initial centroids to guide
k-means clustering toward physically meaningful molecular partitions.

Each strategy takes an (N, 3) array of molecular centroids and the desired
number of clusters, returning an (n_clusters, 3) array of seed positions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _seed_halfplane(
    centroids: np.ndarray, n_clusters: int, **kwargs: Any
) -> np.ndarray:
    """Divide angular space in the PCA plane into sectors.

    Projects points onto the plane spanned by the two leading principal
    components, computes the angle of each point, and assigns points to
    ``n_clusters`` equal angular sectors.  The seed for each sector is the
    mean of its members.

    For n_clusters=2 this naturally produces two half-sphere partitions.
    """
    center = centroids.mean(axis=0)
    shifted = centroids - center

    # PCA via SVD
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    proj = shifted @ vt[:2].T  # project onto first two PCs

    angles = np.arctan2(proj[:, 1], proj[:, 0])  # [-pi, pi]
    bin_edges = np.linspace(-np.pi, np.pi, n_clusters + 1)

    seeds = np.empty((n_clusters, 3))
    for k in range(n_clusters):
        mask = (angles >= bin_edges[k]) & (angles < bin_edges[k + 1])
        if not mask.any():
            # Fallback: place seed at expected angle on unit circle in PCA space
            mid_angle = (bin_edges[k] + bin_edges[k + 1]) / 2
            spread = np.linalg.norm(shifted, axis=1).mean()
            seed_pca = np.array([np.cos(mid_angle), np.sin(mid_angle)]) * spread
            seeds[k] = center + seed_pca @ vt[:2]
        else:
            seeds[k] = centroids[mask].mean(axis=0)
    return seeds


def _seed_pca(
    centroids: np.ndarray, n_clusters: int, **kwargs: Any
) -> np.ndarray:
    """Project onto the first principal component and bin evenly.

    Points are projected onto PC1, the projection range is divided into
    ``n_clusters`` equal-width bins, and the seed for each bin is the mean
    of its members.
    """
    center = centroids.mean(axis=0)
    shifted = centroids - center

    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    proj = shifted @ vt[0]  # scalar projection onto PC1

    bin_edges = np.linspace(proj.min(), proj.max(), n_clusters + 1)

    seeds = np.empty((n_clusters, 3))
    for k in range(n_clusters):
        if k == n_clusters - 1:
            mask = (proj >= bin_edges[k]) & (proj <= bin_edges[k + 1])
        else:
            mask = (proj >= bin_edges[k]) & (proj < bin_edges[k + 1])
        if not mask.any():
            # Fallback: midpoint of bin projected back to 3D
            mid = (bin_edges[k] + bin_edges[k + 1]) / 2
            seeds[k] = center + mid * vt[0]
        else:
            seeds[k] = centroids[mask].mean(axis=0)
    return seeds


def _seed_axis(
    centroids: np.ndarray, n_clusters: int, *, axis: str | None = None, **kwargs: Any
) -> np.ndarray:
    """Sort along a Cartesian axis and bin evenly.

    Parameters
    ----------
    axis : str or None
        One of ``"x"``, ``"y"``, ``"z"``.  If *None* the axis of greatest
        spread is chosen automatically.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis is None:
        # Auto-detect: axis of greatest spread
        spread = centroids.max(axis=0) - centroids.min(axis=0)
        axis_idx = int(np.argmax(spread))
    else:
        axis = axis.lower()
        if axis not in axis_map:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
        axis_idx = axis_map[axis]

    vals = centroids[:, axis_idx]
    bin_edges = np.linspace(vals.min(), vals.max(), n_clusters + 1)

    seeds = np.empty((n_clusters, 3))
    for k in range(n_clusters):
        if k == n_clusters - 1:
            mask = (vals >= bin_edges[k]) & (vals <= bin_edges[k + 1])
        else:
            mask = (vals >= bin_edges[k]) & (vals < bin_edges[k + 1])
        if not mask.any():
            mid = (bin_edges[k] + bin_edges[k + 1]) / 2
            seeds[k] = centroids.mean(axis=0)
            seeds[k, axis_idx] = mid
        else:
            seeds[k] = centroids[mask].mean(axis=0)
    return seeds


def _seed_radial(
    centroids: np.ndarray, n_clusters: int, **kwargs: Any
) -> np.ndarray:
    """Divide the XY angular space around the centre of mass into sectors.

    Computes the angle of each point in the XY plane relative to the
    centre of mass, divides the angular range into ``n_clusters`` equal
    sectors, and returns the mean of each sector as a seed.
    """
    com = centroids.mean(axis=0)
    dx = centroids[:, 0] - com[0]
    dy = centroids[:, 1] - com[1]
    angles = np.arctan2(dy, dx)  # [-pi, pi]

    bin_edges = np.linspace(-np.pi, np.pi, n_clusters + 1)

    seeds = np.empty((n_clusters, 3))
    for k in range(n_clusters):
        mask = (angles >= bin_edges[k]) & (angles < bin_edges[k + 1])
        if not mask.any():
            mid_angle = (bin_edges[k] + bin_edges[k + 1]) / 2
            spread = np.linalg.norm(centroids - com, axis=1).mean()
            seeds[k] = com.copy()
            seeds[k, 0] += spread * np.cos(mid_angle)
            seeds[k, 1] += spread * np.sin(mid_angle)
        else:
            seeds[k] = centroids[mask].mean(axis=0)
    return seeds


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SEEDING_STRATEGIES: Dict[str, Callable[..., np.ndarray]] = {
    "halfplane": _seed_halfplane,
    "pca": _seed_pca,
    "axis": _seed_axis,
    "radial": _seed_radial,
}


def get_strategy(name: str) -> Callable[..., np.ndarray]:
    """Look up a seeding strategy by name.

    Raises
    ------
    ValueError
        If *name* is not a registered strategy.
    """
    try:
        return SEEDING_STRATEGIES[name]
    except KeyError:
        valid = ", ".join(sorted(SEEDING_STRATEGIES))
        raise ValueError(
            f"Unknown seeding strategy '{name}'. Valid strategies: {valid}"
        )


def compute_seeds(
    centroids: np.ndarray, n_clusters: int, strategy: str, **kwargs: Any
) -> np.ndarray:
    """Compute initial centroid seeds for k-means.

    Parameters
    ----------
    centroids : ndarray of shape (N, 3)
        Molecular centroid coordinates.
    n_clusters : int
        Desired number of clusters.
    strategy : str
        Name of a registered seeding strategy.
    **kwargs
        Forwarded to the strategy function (e.g. ``axis="z"``).

    Returns
    -------
    ndarray of shape (n_clusters, 3)
        Seed positions.
    """
    fn = get_strategy(strategy)
    seeds = fn(centroids, n_clusters, **kwargs)
    assert seeds.shape == (n_clusters, 3), (
        f"Strategy '{strategy}' returned shape {seeds.shape}, "
        f"expected ({n_clusters}, 3)"
    )
    return seeds
