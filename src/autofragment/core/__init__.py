# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Core data structures and utilities for autofragment."""

from autofragment.core.atom import CompactAtomArray, atom_dtype
from autofragment.core.bonds import COVALENT_RADII, InterfragmentBond
from autofragment.core.geometry import (
    compute_centroid,
    compute_centroids,
    compute_rmsd,
    kabsch_align,
)
from autofragment.core.graph import SparseBondGraph
from autofragment.core.types import (
    Atom,
    ChemicalSystem,
    Fragment,
    FragmentationScheme,
    FragmentTree,
    Molecule,
    molecules_to_system,
    system_to_molecules,
)

__all__ = [
    "Atom",
    "ChemicalSystem",
    "Molecule",
    "Fragment",
    "FragmentTree",
    "FragmentationScheme",
    "molecules_to_system",
    "system_to_molecules",
    "CompactAtomArray",
    "atom_dtype",
    "InterfragmentBond",
    "COVALENT_RADII",
    "SparseBondGraph",
    "compute_centroid",
    "compute_centroids",
    "compute_rmsd",
    "kabsch_align",
]
