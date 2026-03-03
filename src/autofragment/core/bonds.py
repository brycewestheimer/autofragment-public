# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Bond detection and interfragment bond representation.

This module provides:
- InterfragmentBond: Represents a bond between two fragments
- COVALENT_RADII: Standard covalent radii for common elements
- Bond detection utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from autofragment.core.types import Atom


class InterfragmentBond:
    """
    Represents a bond between two fragments.

    This class tracks bonds that cross fragment boundaries, which need special
    handling during fragment-based calculations (e.g., hydrogen capping).

    Attributes
    ----------
    fragment1_id : str
        ID of the first fragment.
    atom1_index : int
        Local atom index within fragment1 (0-based).
    fragment2_id : str
        ID of the second fragment.
    atom2_index : int
        Local atom index within fragment2 (0-based).
    bond_order : float
        Bond order (1=single, 2=double, 3=triple, 1.5=aromatic).
    metadata : Dict[str, Any]
        Additional bond information (e.g., type, inferred flag).
    """

    def __init__(
        self,
        fragment1_id: str,
        atom1_index: int,
        fragment2_id: str,
        atom2_index: int,
        bond_order: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a new InterfragmentBond instance."""
        self.fragment1_id = fragment1_id
        self.atom1_index = atom1_index
        self.fragment2_id = fragment2_id
        self.atom2_index = atom2_index
        self.bond_order = bond_order
        self.metadata = metadata or {}

    def involves_fragment(self, fragment_id: str) -> bool:
        """Check if this bond involves the specified fragment."""
        return fragment_id == self.fragment1_id or fragment_id == self.fragment2_id

    def get_partner_fragment(self, fragment_id: str) -> Optional[str]:
        """Get the ID of the fragment on the other side of this bond."""
        if fragment_id == self.fragment1_id:
            return self.fragment2_id
        elif fragment_id == self.fragment2_id:
            return self.fragment1_id
        return None

    def get_atom_index_in_fragment(self, fragment_id: str) -> Optional[int]:
        """Get the atom index for this bond in the specified fragment."""
        if fragment_id == self.fragment1_id:
            return self.atom1_index
        elif fragment_id == self.fragment2_id:
            return self.atom2_index
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fragment1_id": self.fragment1_id,
            "atom1_index": self.atom1_index,
            "fragment2_id": self.fragment2_id,
            "atom2_index": self.atom2_index,
            "bond_order": self.bond_order,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InterfragmentBond:
        """Create from dictionary (for JSON deserialization)."""
        return cls(
            fragment1_id=data["fragment1_id"],
            atom1_index=data["atom1_index"],
            fragment2_id=data["fragment2_id"],
            atom2_index=data["atom2_index"],
            bond_order=data.get("bond_order", 1.0),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"InterfragmentBond({self.fragment1_id}[{self.atom1_index}] - "
            f"{self.fragment2_id}[{self.atom2_index}], order={self.bond_order})"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether this object is equal to another object."""
        if not isinstance(other, InterfragmentBond):
            return False
        # Bond is the same regardless of direction
        return (
            self.fragment1_id == other.fragment1_id
            and self.atom1_index == other.atom1_index
            and self.fragment2_id == other.fragment2_id
            and self.atom2_index == other.atom2_index
        ) or (
            self.fragment1_id == other.fragment2_id
            and self.atom1_index == other.atom2_index
            and self.fragment2_id == other.fragment1_id
            and self.atom2_index == other.atom1_index
        )


# Standard covalent radii for common elements (in Angstroms)
# Used for automatic bond detection
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Si": 1.11,
    "B": 0.84,
    "Al": 1.21,
    "Li": 1.28,
    "Na": 1.66,
    "K": 2.03,
    "Mg": 1.41,
    "Ca": 1.76,
    "Fe": 1.32,
    "Zn": 1.22,
    "Cu": 1.32,
    "Ni": 1.24,
    "He": 0.28,
    "Ne": 0.58,
    "Ar": 1.06,
}


def get_covalent_radius(element: str) -> float:
    """
    Get the covalent radius for an element.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'H', 'O').

    Returns
    -------
    float
        Covalent radius in Angstroms.

    Raises
    ------
    ValueError
        If element is not recognized.
    """
    elem_upper = element.strip().capitalize()
    if elem_upper not in COVALENT_RADII:
        raise ValueError(f"Unknown element: {element}")
    return COVALENT_RADII[elem_upper]


def is_bonded(
    atom1_symbol: str,
    atom1_coords: Tuple[float, float, float],
    atom2_symbol: str,
    atom2_coords: Tuple[float, float, float],
    tolerance: float = 0.4,
) -> bool:
    """
    Determine if two atoms are bonded based on distance.

    Two atoms are considered bonded if their distance is less than the sum
    of their covalent radii plus a tolerance factor.

    Parameters
    ----------
    atom1_symbol : str
        Element symbol of first atom.
    atom1_coords : tuple
        (x, y, z) coordinates of first atom in Angstroms.
    atom2_symbol : str
        Element symbol of second atom.
    atom2_coords : tuple
        (x, y, z) coordinates of second atom in Angstroms.
    tolerance : float, optional
        Additional distance tolerance in Angstroms. Default is 0.4.

    Returns
    -------
    bool
        True if atoms are bonded, False otherwise.
    """
    try:
        r1 = get_covalent_radius(atom1_symbol)
        r2 = get_covalent_radius(atom2_symbol)
    except ValueError:
        # Unknown element, can't determine bonding
        return False

    coords1 = np.array(atom1_coords)
    coords2 = np.array(atom2_coords)
    distance = float(np.linalg.norm(coords1 - coords2))

    max_bond_distance = r1 + r2 + tolerance
    return distance < max_bond_distance


def infer_bonds_from_atoms(
    atoms: "List[Atom]",
    tolerance: float = 0.4,
) -> List[Dict[str, Any]]:
    """Infer bonds from a list of Atom objects using covalent radii.

    For systems with more than 1000 atoms, a ``scipy.spatial.cKDTree``
    is used for neighbour detection to avoid the O(N^2) pairwise loop.
    If *scipy* is not available the brute-force approach is used as a
    fallback.

    Parameters
    ----------
    atoms : list of Atom
        Atoms with ``.symbol`` and ``.coords`` attributes.
    tolerance : float, optional
        Additional distance tolerance in Angstroms (default 0.4).

    Returns
    -------
    list of dict
        Bond dicts with keys ``atom1``, ``atom2``, ``order``.
    """
    n = len(atoms)
    if n < 2:
        return []

    bonds: List[Dict[str, Any]] = []

    if n > 1000:
        try:
            from scipy.spatial import cKDTree  # type: ignore[import-untyped]

            coords = np.array([a.coords for a in atoms])
            max_radius = max(COVALENT_RADII.values())
            cutoff = max_radius * 2 + tolerance
            tree = cKDTree(coords)
            pairs = tree.query_pairs(cutoff)
            for i, j in pairs:
                if is_bonded(
                    atoms[i].symbol,
                    tuple(atoms[i].coords),
                    atoms[j].symbol,
                    tuple(atoms[j].coords),
                    tolerance=tolerance,
                ):
                    bonds.append({"atom1": i, "atom2": j, "order": 1.0})
            return bonds
        except ImportError:
            pass  # fall through to brute-force

    for i in range(n):
        for j in range(i + 1, n):
            if is_bonded(
                atoms[i].symbol,
                tuple(atoms[i].coords),
                atoms[j].symbol,
                tuple(atoms[j].coords),
                tolerance=tolerance,
            ):
                bonds.append({"atom1": i, "atom2": j, "order": 1.0})

    return bonds
