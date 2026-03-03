# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric utilities for isolated molecular fragments.

This module provides functions for:
- Computing centroids
- Computing RMSD between structures
- Kabsch alignment

These helpers operate on isolated Molecule objects. Convert from
ChemicalSystem explicitly when working with full systems.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from autofragment.core.types import Atom, Molecule, molecule_to_coords


def compute_centroid(coords: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of a set of coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array with shape (N, 3).

    Returns
    -------
    np.ndarray
        Centroid coordinates with shape (3,).
    """
    return np.mean(coords, axis=0)


def compute_centroids(molecules: Sequence[Molecule]) -> np.ndarray:
    """
    Compute centroids for a list of isolated molecules.

    Parameters
    ----------
    molecules : Sequence[Molecule]
        List of molecules (each molecule is a list of Atoms).

    Returns
    -------
    np.ndarray
        Centroids array with shape (N_molecules, 3).
    """
    centroids = []
    for mol in molecules:
        coords = molecule_to_coords(mol)
        centroids.append(coords.mean(axis=0))
    return np.stack(centroids, axis=0)


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute the root-mean-square deviation between two coordinate sets.

    Parameters
    ----------
    coords1 : np.ndarray
        First coordinates array with shape (N, 3).
    coords2 : np.ndarray
        Second coordinates array with shape (N, 3).

    Returns
    -------
    float
        RMSD value in the same units as the input coordinates.
    """
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def molecule_rmsd(mol1: Molecule, mol2: Molecule) -> float:
    """
    Compute RMSD between two molecules.

    Parameters
    ----------
    mol1 : Molecule
        First molecule.
    mol2 : Molecule
        Second molecule.

    Returns
    -------
    float
        RMSD value in Angstroms.
    """
    coords1 = molecule_to_coords(mol1)
    coords2 = molecule_to_coords(mol2)
    return compute_rmsd(coords1, coords2)


def hybrid_distance(mol1: Molecule, mol2: Molecule, alpha: float = 0.5) -> float:
    """
    Compute a hybrid distance combining centroid distance and RMSD.

    Parameters
    ----------
    mol1 : Molecule
        First molecule.
    mol2 : Molecule
        Second molecule.
    alpha : float, optional
        Weight for centroid distance (0 to 1). Default is 0.5.
        The RMSD weight is (1 - alpha).

    Returns
    -------
    float
        Hybrid distance value.
    """
    coords1 = molecule_to_coords(mol1)
    coords2 = molecule_to_coords(mol2)

    cent1 = coords1.mean(axis=0)
    cent2 = coords2.mean(axis=0)
    cent_dist = float(np.linalg.norm(cent1 - cent2))

    rmsd = compute_rmsd(coords1, coords2)

    return alpha * cent_dist + (1 - alpha) * rmsd


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute the optimal rotation matrix to align P onto Q using Kabsch algorithm.

    Parameters
    ----------
    P : np.ndarray
        Coordinates to rotate, shape (N, 3), centered at origin.
    Q : np.ndarray
        Target coordinates, shape (N, 3), centered at origin.

    Returns
    -------
    np.ndarray
        Rotation matrix with shape (3, 3).
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def kabsch_align(
    reference: Sequence[Molecule],
    target: Sequence[Molecule],
) -> List[Molecule]:
    """
    Apply Kabsch alignment to target molecules based on centroids.

    This aligns the target molecules onto the reference molecules by
    computing the optimal rotation based on molecular centroids.

    Parameters
    ----------
    reference : Sequence[Molecule]
        Reference molecules to align to.
    target : Sequence[Molecule]
        Target molecules to be aligned.

    Returns
    -------
    List[Molecule]
        Aligned target molecules.
    """
    ref_centroids = compute_centroids(reference)
    tgt_centroids = compute_centroids(target)

    ref_centered = ref_centroids - ref_centroids.mean(axis=0)
    tgt_centered = tgt_centroids - tgt_centroids.mean(axis=0)

    R = kabsch_rotation(tgt_centered, ref_centered)

    ref_mean = ref_centroids.mean(axis=0)
    tgt_mean = tgt_centroids.mean(axis=0)

    aligned_molecules: List[Molecule] = []
    for mol in target:
        coords = molecule_to_coords(mol)
        aligned_coords = (R @ (coords - tgt_mean).T).T + ref_mean
        aligned_mol = [
            Atom(symbol=atom.symbol, coords=aligned_coords[i])
            for i, atom in enumerate(mol)
        ]
        aligned_molecules.append(aligned_mol)

    return aligned_molecules
