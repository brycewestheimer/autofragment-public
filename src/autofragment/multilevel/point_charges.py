# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Point charge embedding for electrostatic QM/MM.

This module provides point charge generation and formatting
for electrostatic embedding in QM/MM calculations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PointCharge:
    """A point charge for electrostatic embedding.

    Attributes:
        position: Cartesian coordinates (3,) in Angstrom.
        charge: Partial charge in atomic units (e).
        atom_index: Original MM atom index (optional).
        element: Element symbol for identification.
    """

    position: np.ndarray
    charge: float
    atom_index: Optional[int] = None
    element: Optional[str] = None

    def distance_to(self, other_position: np.ndarray) -> float:
        """Calculate distance to another point."""
        return float(np.linalg.norm(self.position - other_position))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position": self.position.tolist(),
            "charge": self.charge,
            "atom_index": self.atom_index,
            "element": self.element,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointCharge":
        """Create from dictionary."""
        return cls(
            position=np.array(data["position"]),
            charge=data["charge"],
            atom_index=data.get("atom_index"),
            element=data.get("element"),
        )


# Common partial charges for amino acid atoms (AMBER-like)
AMINO_ACID_CHARGES: Dict[str, Dict[str, float]] = {
    "backbone": {
        "N": -0.4157,
        "H": 0.2719,
        "CA": 0.0337,
        "HA": 0.0823,
        "C": 0.5973,
        "O": -0.5679,
    },
    "ALA": {"CB": -0.1825, "HB1": 0.0603, "HB2": 0.0603, "HB3": 0.0603},
    "GLY": {"HA2": 0.0698, "HA3": 0.0698},
}


# Default partial charges by element (Gasteiger-like fallback)
DEFAULT_CHARGES: Dict[str, float] = {
    "H": 0.15,
    "C": -0.10,
    "N": -0.30,
    "O": -0.35,
    "S": -0.20,
    "P": 0.35,
    "Na": 1.0,
    "Cl": -1.0,
    "Mg": 2.0,
    "Ca": 2.0,
    "Zn": 2.0,
    "Fe": 2.0,
}


def get_element_charge(element: str) -> float:
    """Get default partial charge for an element.

    Args:
        element: Element symbol.

    Returns:
        Partial charge in atomic units.
    """
    return DEFAULT_CHARGES.get(element, 0.0)


class PointChargeEmbedding:
    """Generate and manage point charge arrays for QM/MM.

    This class generates point charges from MM atoms for use
    in electrostatic embedding calculations.

    Attributes:
        charge_scheme: Source of charges ("default", "amber", etc.).
        exclude_link_atoms: Whether to exclude link atom positions.
        charge_shift_scheme: Near-boundary charge handling.
        shift_radius: Radius for charge redistribution.
    """

    def __init__(
        self,
        charge_scheme: str = "default",
        exclude_link_atoms: bool = True,
        charge_shift_scheme: Optional[str] = None,
        shift_radius: float = 2.0,
    ):
        """Initialize point charge embedding.

        Args:
            charge_scheme: "default", "amber", "charmm", or "fixed".
            exclude_link_atoms: Whether to exclude MM atoms with links.
            charge_shift_scheme: "div" (divide), "shift", or None.
            shift_radius: Cutoff for charge redistribution.
        """
        self.charge_scheme = charge_scheme
        self.exclude_link_atoms = exclude_link_atoms
        self.charge_shift_scheme = charge_shift_scheme
        self.shift_radius = shift_radius

    def generate_charges(
        self,
        coords: np.ndarray,
        elements: List[str],
        mm_atoms: Set[int],
        link_mm_atoms: Optional[Set[int]] = None,
        custom_charges: Optional[Dict[int, float]] = None,
    ) -> List[PointCharge]:
        """Generate point charges for MM atoms.

        Args:
            coords: (N, 3) atomic coordinates.
            elements: Element symbols for all atoms.
            mm_atoms: Indices of MM atoms.
            link_mm_atoms: MM atoms replaced by link atoms.
            custom_charges: Custom charges by atom index.

        Returns:
            List of PointCharge objects.
        """
        charges = []

        # Determine which MM atoms to exclude
        excluded = link_mm_atoms if (self.exclude_link_atoms and link_mm_atoms) else set()

        for mm_idx in mm_atoms:
            if mm_idx in excluded:
                continue

            # Get charge
            if custom_charges and mm_idx in custom_charges:
                charge = custom_charges[mm_idx]
            else:
                charge = get_element_charge(elements[mm_idx])

            pc = PointCharge(
                position=coords[mm_idx].copy(),
                charge=charge,
                atom_index=mm_idx,
                element=elements[mm_idx],
            )
            charges.append(pc)

        # Apply charge redistribution near link atoms if requested
        if self.charge_shift_scheme and link_mm_atoms:
            charges = self._apply_charge_redistribution(
                charges, coords, link_mm_atoms
            )

        return charges

    def _apply_charge_redistribution(
        self,
        charges: List[PointCharge],
        coords: np.ndarray,
        link_mm_atoms: Set[int],
    ) -> List[PointCharge]:
        """Apply charge redistribution near link atom positions.

        This avoids overpolarization of the QM region by nearby charges.

        Args:
            charges: List of point charges.
            coords: Original atomic coordinates.
            link_mm_atoms: MM atoms that have been replaced.

        Returns:
            Modified list of point charges.
        """
        if not self.charge_shift_scheme:
            return charges

        result = []
        link_positions = [coords[idx] for idx in link_mm_atoms]

        for pc in charges:
            # Check distance to any link atom position
            if link_positions:
                distances = [float(np.linalg.norm(pc.position - lp)) for lp in link_positions]
                min_dist: float = min(distances)
            else:
                min_dist = float("inf")

            if min_dist < self.shift_radius:
                if self.charge_shift_scheme == "div":
                    # Reduce charge based on proximity
                    scale = min_dist / self.shift_radius
                    pc = PointCharge(
                        position=pc.position,
                        charge=float(pc.charge * scale),
                        atom_index=pc.atom_index,
                        element=pc.element,
                    )
                elif self.charge_shift_scheme == "shift":
                    # Move charge to shift radius boundary
                    direction = pc.position - link_positions[0]
                    direction = direction / np.linalg.norm(direction)
                    new_position = link_positions[0] + self.shift_radius * direction
                    pc = PointCharge(
                        position=new_position,
                        charge=pc.charge,
                        atom_index=pc.atom_index,
                        element=pc.element,
                    )

            result.append(pc)

        return result

    def to_arrays(
        self,
        charges: List[PointCharge],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get charges as numpy arrays.

        Args:
            charges: List of point charges.

        Returns:
            Tuple of (positions (N,3), charges (N,)).
        """
        if not charges:
            return np.array([]).reshape(0, 3), np.array([])

        positions = np.array([pc.position for pc in charges])
        charge_values = np.array([pc.charge for pc in charges])

        return positions, charge_values

    def to_gamess_format(self, charges: List[PointCharge]) -> str:
        """Format charges for GAMESS $EFRAG group.

        Args:
            charges: List of point charges.

        Returns:
            GAMESS-formatted string.
        """
        lines = [" $EFRAG"]
        lines.append(" COORD=CART")
        lines.append(f" NCHARG={len(charges)}")

        for pc in charges:
            x, y, z = pc.position
            lines.append(f"  {x:12.6f} {y:12.6f} {z:12.6f} {pc.charge:10.6f}")

        lines.append(" $END")
        return "\n".join(lines)

    def to_gaussian_format(self, charges: List[PointCharge]) -> str:
        """Format charges for Gaussian input.

        Args:
            charges: List of point charges.

        Returns:
            Gaussian-formatted string for charge input.
        """
        lines = []
        for pc in charges:
            x, y, z = pc.position
            lines.append(f"{x:15.8f}  {y:15.8f}  {z:15.8f}  {pc.charge:12.8f}")

        return "\n".join(lines)

    def total_charge(self, charges: List[PointCharge]) -> float:
        """Get total charge of all point charges."""
        return sum(pc.charge for pc in charges)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize embedding configuration."""
        return {
            "charge_scheme": self.charge_scheme,
            "exclude_link_atoms": self.exclude_link_atoms,
            "charge_shift_scheme": self.charge_shift_scheme,
            "shift_radius": self.shift_radius,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointChargeEmbedding":
        """Create from dictionary."""
        return cls(
            charge_scheme=data.get("charge_scheme", "default"),
            exclude_link_atoms=data.get("exclude_link_atoms", True),
            charge_shift_scheme=data.get("charge_shift_scheme"),
            shift_radius=data.get("shift_radius", 2.0),
        )


def generate_simple_charge_array(
    coords: np.ndarray,
    elements: List[str],
    mm_atoms: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple convenience function for generating charge arrays.

    Args:
        coords: Atomic coordinates.
        elements: Element symbols.
        mm_atoms: Indices of MM atoms.

    Returns:
        Tuple of (positions, charges) arrays.
    """
    embedding = PointChargeEmbedding()
    charges = embedding.generate_charges(coords, elements, mm_atoms)
    return embedding.to_arrays(charges)
