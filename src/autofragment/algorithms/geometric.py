# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric plane-based partitioning for molecular clusters.

This module provides geometric partitioning based on planes through
the cluster center, useful for symmetric systems like water clusters.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def find_center(coords: np.ndarray, method: str = "center_of_mass") -> np.ndarray:
    """
    Find the center of the cluster.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array with shape (N, 3).
    method : str, optional
        Center calculation method. Currently only "center_of_mass"
        and "geometric_center" are supported (both equivalent for
        equal-mass particles). Default is "center_of_mass".

    Returns
    -------
    np.ndarray
        Center coordinates with shape (3,).
    """
    if method in ("center_of_mass", "geometric_center"):
        return np.mean(coords, axis=0)
    raise ValueError(f"Unknown center method: {method}")


def partition_by_planes(
    coords: np.ndarray,
    n_groups: int,
    center_method: str = "center_of_mass",
) -> np.ndarray:
    """Partition coordinates into groups using angular slices.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array with shape (N, 3).
    n_groups : int
        Number of groups (angular slices around the z-axis).
    center_method : str, optional
        Method for computing the center. Default is "center_of_mass".

    Returns
    -------
    np.ndarray
        Labels with shape (N,) in range [0, n_groups-1].
    """
    center = find_center(coords, method=center_method)
    rel_coords = coords - center

    if n_groups <= 0:
        raise ValueError("n_groups must be positive")

    # Angular slices around z-axis
    angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
    labels = np.floor((angles + np.pi) / (2 * np.pi / n_groups)).astype(int)
    return np.clip(labels, 0, n_groups - 1)


def partition_by_planes_tiered(
    coords: np.ndarray,
    n_primary: int,
    n_secondary: int = 1,
    n_tertiary: int = 1,
    center_method: str = "center_of_mass",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition coordinates into hierarchical regions using geometric planes.

    This method uses:
    - Primary: Angular slices around the z-axis (like pizza slices)
    - Secondary: Horizontal slices along the z-axis
    - Tertiary: Slices along the y-axis

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array with shape (N, 3).
    n_primary : int
        Number of primary groups (angular slices).
    n_secondary : int, optional
        Number of secondary groups (z-axis slices). Default is 1.
    n_tertiary : int, optional
        Number of tertiary groups (y-axis slices). Default is 1.
    center_method : str, optional
        Method for computing the center. Default is "center_of_mass".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (primary_labels, secondary_labels, tertiary_labels)
        Each array has shape (N,) with integer labels.
    """
    center = find_center(coords, method=center_method)
    rel_coords = coords - center

    # Primary: angular slices around z-axis
    angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
    primary_labels = np.floor(
        (angles + np.pi) / (2 * np.pi / n_primary)
    ).astype(int)
    primary_labels = np.clip(primary_labels, 0, n_primary - 1)

    # Secondary: z-coordinate slices
    secondary_labels = np.zeros(len(coords), dtype=int)
    if n_secondary > 1:
        z_sorted = np.argsort(rel_coords[:, 2])
        per_sec = len(coords) // n_secondary
        for i in range(n_secondary):
            start = i * per_sec
            end = (i + 1) * per_sec if i < n_secondary - 1 else len(coords)
            secondary_labels[z_sorted[start:end]] = i

    # Tertiary: y-coordinate slices
    tertiary_labels = np.zeros(len(coords), dtype=int)
    if n_tertiary > 1:
        y_sorted = np.argsort(rel_coords[:, 1])
        per_ter = len(coords) // n_tertiary
        for i in range(n_tertiary):
            start = i * per_ter
            end = (i + 1) * per_ter if i < n_tertiary - 1 else len(coords)
            tertiary_labels[y_sorted[start:end]] = i

    return primary_labels, secondary_labels, tertiary_labels
