# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Advanced link atom positioning for QM/MM boundaries.

This module provides sophisticated link atom positioning using g-factor
approach and supporting different element pairs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Standard bond lengths in Angstrom
BOND_LENGTHS: Dict[Tuple[str, str], float] = {
    ("C", "H"): 1.09,
    ("C", "C"): 1.54,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "S"): 1.82,
    ("N", "H"): 1.01,
    ("N", "C"): 1.47,
    ("O", "H"): 0.96,
    ("S", "H"): 1.34,
}


def get_bond_length(element1: str, element2: str) -> float:
    """Get typical bond length between two elements.

    Args:
        element1: First element symbol.
        element2: Second element symbol.

    Returns:
        Bond length in Angstrom.
    """
    key1 = (element1, element2)
    key2 = (element2, element1)

    if key1 in BOND_LENGTHS:
        return BOND_LENGTHS[key1]
    elif key2 in BOND_LENGTHS:
        return BOND_LENGTHS[key2]
    else:
        # Default to carbon-carbon
        return 1.54


def calculate_g_factor(
    qm_element: str,
    mm_element: str,
    link_element: str = "H",
) -> float:
    """Calculate g-factor for link atom positioning.

    The g-factor determines the position of the link atom along
    the QM-MM bond vector.

    Args:
        qm_element: Element of QM atom.
        mm_element: Element of MM atom.
        link_element: Element of link atom (usually H).

    Returns:
        g-factor (ratio of link bond length to original bond).

    Example:
        >>> g = calculate_g_factor("C", "C", "H")
        >>> round(g, 3)
        0.708
    """
    # Get bond lengths
    link_bond = get_bond_length(qm_element, link_element)
    original_bond = get_bond_length(qm_element, mm_element)

    return link_bond / original_bond


def position_link_atom_gfactor(
    qm_position: np.ndarray,
    mm_position: np.ndarray,
    g_factor: float,
) -> np.ndarray:
    """Position link atom using g-factor approach.

    R_link = R_QM + g * (R_MM - R_QM)

    Args:
        qm_position: Position of QM atom (3,).
        mm_position: Position of MM atom (3,).
        g_factor: Scaling factor for positioning.

    Returns:
        Position of link atom (3,).
    """
    bond_vector = mm_position - qm_position
    return qm_position + g_factor * bond_vector


def position_link_atom_fixed_distance(
    qm_position: np.ndarray,
    mm_position: np.ndarray,
    distance: float = 1.09,
) -> np.ndarray:
    """Position link atom at fixed distance from QM atom.

    Args:
        qm_position: Position of QM atom (3,).
        mm_position: Position of MM atom (3,).
        distance: Distance from QM atom in Angstrom.

    Returns:
        Position of link atom (3,).
    """
    bond_vector = mm_position - qm_position
    bond_length = np.linalg.norm(bond_vector)
    unit_vector = bond_vector / bond_length

    return qm_position + distance * unit_vector


@dataclass
class LinkAtomInfo:
    """Complete information about a link atom.

    Contains all data needed for link atom handling including
    position, connected atoms, and force redistribution.

    Attributes:
        qm_atom_index: Index of QM atom link attaches to.
        mm_atom_index: Index of MM atom being replaced.
        element: Element symbol of link atom.
        position: Cartesian coordinates of link atom.
        g_factor: Positioning g-factor.
        force_scale: Scale factor for force redistribution.
    """

    qm_atom_index: int
    mm_atom_index: int
    element: str = "H"
    position: Optional[np.ndarray] = None
    g_factor: float = 0.709
    force_scale: float = 1.0

    def compute_position(
        self,
        qm_coords: np.ndarray,
        mm_coords: np.ndarray,
    ) -> np.ndarray:
        """Compute and store link atom position.

        Args:
            qm_coords: Coordinates of QM atom.
            mm_coords: Coordinates of MM atom.

        Returns:
            Link atom position.
        """
        self.position = position_link_atom_gfactor(qm_coords, mm_coords, self.g_factor)
        return self.position

    def redistribute_force(
        self,
        link_force: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Redistribute forces from link atom to real atoms.

        Args:
            link_force: Force on link atom (3,).

        Returns:
            Tuple of (QM atom force, MM atom force).
        """
        # Simple redistribution based on g-factor
        qm_force = (1 - self.g_factor) * link_force
        mm_force = self.g_factor * link_force

        return qm_force, mm_force


def create_link_atoms_for_cut_bonds(
    coords: np.ndarray,
    elements: List[str],
    cut_bonds: List[Tuple[int, int]],
    link_element: str = "H",
) -> List[LinkAtomInfo]:
    """Create link atoms for all cut bonds.

    Args:
        coords: (N, 3) array of atomic coordinates.
        elements: List of element symbols.
        cut_bonds: List of (qm_idx, mm_idx) tuples.
        link_element: Element for link atoms.

    Returns:
        List of configured LinkAtomInfo objects.
    """
    link_atoms = []

    for qm_idx, mm_idx in cut_bonds:
        qm_element = elements[qm_idx]
        mm_element = elements[mm_idx]

        g_factor = calculate_g_factor(qm_element, mm_element, link_element)

        link_info = LinkAtomInfo(
            qm_atom_index=qm_idx,
            mm_atom_index=mm_idx,
            element=link_element,
            g_factor=g_factor,
        )

        link_info.compute_position(coords[qm_idx], coords[mm_idx])
        link_atoms.append(link_info)

    return link_atoms


def validate_link_atoms(
    link_atoms: List[LinkAtomInfo],
    coords: np.ndarray,
    min_distance: float = 0.8,
) -> List[str]:
    """Validate link atom positions.

    Args:
        link_atoms: List of link atoms to validate.
        coords: Original atomic coordinates.
        min_distance: Minimum allowed distance in Angstrom.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings = []

    for i, link in enumerate(link_atoms):
        if link.position is None:
            warnings.append(f"Link atom {i}: position not computed")
            continue

        # Check distance from QM atom
        qm_dist = np.linalg.norm(link.position - coords[link.qm_atom_index])
        if qm_dist < min_distance:
            warnings.append(
                f"Link atom {i}: too close to QM atom ({qm_dist:.3f} Å)"
            )

        # Check g-factor is reasonable
        if link.g_factor < 0.5 or link.g_factor > 1.0:
            warnings.append(
                f"Link atom {i}: unusual g-factor ({link.g_factor:.3f})"
            )

    return warnings
