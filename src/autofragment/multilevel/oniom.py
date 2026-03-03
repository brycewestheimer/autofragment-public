# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""ONIOM-style layer specification and input generation.

This module provides ONIOM-specific functionality including parsing
of ONIOM specification strings and generation of input for various
quantum chemistry programs.

Example:
    >>> scheme = ONIOMScheme.from_string("ONIOM(B3LYP/6-31G*:AMBER)")
    >>> scheme.to_gaussian_input()
    'ONIOM(B3LYP/6-31G*:AMBER)'
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from autofragment.multilevel.layers import (
    ComputationalLayer,
    EmbeddingType,
    LayerType,
    MultiLevelScheme,
)


@dataclass
class MethodBasisPair:
    """A method/basis set pair parsed from an ONIOM string.

    Attributes:
        method: Computational method name.
        basis: Basis set name (None for MM methods).
        is_mm: Whether this is a molecular mechanics method.
    """

    method: str
    basis: Optional[str] = None
    is_mm: bool = False

    def __str__(self) -> str:
        """Return a user-facing string representation."""
        if self.basis:
            return f"{self.method}/{self.basis}"
        return self.method


def parse_method_basis(spec: str) -> MethodBasisPair:
    """Parse a method/basis specification string.

    Args:
        spec: String like "B3LYP/6-31G*" or "AMBER".

    Returns:
        MethodBasisPair with parsed components.
    """
    # Known MM force fields (add more as needed)
    mm_methods = {
        "UFF", "AMBER", "DREIDING", "GAFF", "OPLS", "CHARMM",
        "MM3", "MM4", "MMFF94", "AMOEBA",
    }

    if "/" in spec:
        parts = spec.split("/", 1)
        return MethodBasisPair(method=parts[0], basis=parts[1], is_mm=False)
    else:
        is_mm = spec.upper() in mm_methods
        return MethodBasisPair(method=spec, basis=None, is_mm=is_mm)


class ONIOMScheme(MultiLevelScheme):
    """ONIOM-specific multi-level scheme.

    Supports standard ONIOM notation like ONIOM(high:medium:low) and
    provides convenient methods for creating and manipulating ONIOM
    calculations.

    Attributes:
        high_method: Method for high-level region.
        high_basis: Basis set for high-level region.
        medium_method: Method for medium-level region (3-layer only).
        medium_basis: Basis set for medium-level region.
        low_method: Method for low-level region.
        low_basis: Basis set for low-level region (None for MM).

    Example:
        >>> scheme = ONIOMScheme(
        ...     high_method="B3LYP",
        ...     high_basis="6-31G*",
        ...     low_method="UFF"
        ... )
        >>> scheme.n_layers
        2
    """

    def __init__(
        self,
        high_method: str,
        high_basis: str,
        low_method: str = "UFF",
        low_basis: Optional[str] = None,
        medium_method: Optional[str] = None,
        medium_basis: Optional[str] = None,
        embedding: EmbeddingType = EmbeddingType.ELECTROSTATIC,
        name: Optional[str] = None,
    ):
        """Initialize an ONIOM scheme.

        Args:
            high_method: Method for high-level layer.
            high_basis: Basis set for high-level layer.
            low_method: Method for low-level layer (default "UFF").
            low_basis: Basis set for low-level layer (None for MM).
            medium_method: Method for medium layer (optional, 3-layer).
            medium_basis: Basis set for medium layer.
            embedding: Embedding type for QM/MM interactions.
            name: Scheme name (auto-generated if None).
        """
        # Store method/basis info
        self.high_method = high_method
        self.high_basis = high_basis
        self.low_method = low_method
        self.low_basis = low_basis
        self.medium_method = medium_method
        self.medium_basis = medium_basis

        # Generate name if not provided
        if name is None:
            name = self._generate_name()

        # Initialize parent
        super().__init__(
            name=name,
            scheme_type="oniom",
            embedding_type=embedding,
        )

        # Create placeholder layers (atoms added later)
        self._create_layers()

    def _generate_name(self) -> str:
        """Generate scheme name from methods."""
        if self.medium_method:
            return f"ONIOM({self.high_method}/{self.high_basis}:{self.medium_method}/{self.medium_basis}:{self.low_method})"
        else:
            return f"ONIOM({self.high_method}/{self.high_basis}:{self.low_method})"

    def _create_layers(self) -> None:
        """Create computational layers based on method specs."""
        # High layer (always QM)
        high_layer = ComputationalLayer(
            name="high",
            layer_type=LayerType.HIGH,
            method=self.high_method,
            basis_set=self.high_basis,
        )
        self.layers.append(high_layer)

        # Medium layer (if 3-layer)
        if self.medium_method:
            medium_layer = ComputationalLayer(
                name="medium",
                layer_type=LayerType.MEDIUM,
                method=self.medium_method,
                basis_set=self.medium_basis,
            )
            self.layers.append(medium_layer)

        # Low layer (QM or MM)
        if self.low_basis:
            # QM low layer
            low_layer = ComputationalLayer(
                name="low",
                layer_type=LayerType.LOW,
                method=self.low_method,
                basis_set=self.low_basis,
            )
        else:
            # MM low layer
            low_layer = ComputationalLayer(
                name="low",
                layer_type=LayerType.MM,
                method=self.low_method,
            )
        self.layers.append(low_layer)

    @property
    def is_three_layer(self) -> bool:
        """Check if this is a 3-layer ONIOM scheme."""
        return self.medium_method is not None

    @classmethod
    def from_string(cls, spec: str) -> "ONIOMScheme":
        """Parse an ONIOM specification string.

        Supports two-layer and three-layer ONIOM specifications in
        standard notation.

        Args:
            spec: ONIOM specification like "ONIOM(B3LYP/6-31G*:UFF)" or
                "ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G*:AMBER)".

        Returns:
            Configured ONIOMScheme instance.

        Raises:
            ValueError: If the specification cannot be parsed.

        Example:
            >>> scheme = ONIOMScheme.from_string("ONIOM(B3LYP/6-31G*:UFF)")
            >>> scheme.high_method
            'B3LYP'
        """
        # Remove whitespace
        spec = spec.strip()

        # Parse "ONIOM(...)" format
        match = re.match(r"ONIOM\s*\((.+)\)", spec, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Invalid ONIOM specification: {spec}. "
                f"Expected format: ONIOM(high:low) or ONIOM(high:medium:low)"
            )

        inner = match.group(1)

        # Split by colon, but be careful with method names containing colons
        # (unlikely but possible)
        layer_specs = inner.split(":")

        if len(layer_specs) == 2:
            return cls._from_two_layer(layer_specs[0], layer_specs[1])
        elif len(layer_specs) == 3:
            return cls._from_three_layer(
                layer_specs[0], layer_specs[1], layer_specs[2]
            )
        else:
            raise ValueError(
                f"ONIOM requires 2 or 3 layers, got {len(layer_specs)}: {spec}"
            )

    @classmethod
    def _from_two_layer(cls, high_spec: str, low_spec: str) -> "ONIOMScheme":
        """Create from two-layer specification."""
        high = parse_method_basis(high_spec.strip())
        low = parse_method_basis(low_spec.strip())

        if high.is_mm:
            raise ValueError(f"High layer cannot be MM method: {high_spec}")
        if high.basis is None:
            raise ValueError(f"High layer requires a basis set: {high_spec}")

        return cls(
            high_method=high.method,
            high_basis=high.basis,
            low_method=low.method,
            low_basis=low.basis,
        )

    @classmethod
    def _from_three_layer(
        cls, high_spec: str, medium_spec: str, low_spec: str
    ) -> "ONIOMScheme":
        """Create from three-layer specification."""
        high = parse_method_basis(high_spec.strip())
        medium = parse_method_basis(medium_spec.strip())
        low = parse_method_basis(low_spec.strip())

        if high.is_mm:
            raise ValueError(f"High layer cannot be MM method: {high_spec}")
        if medium.is_mm:
            raise ValueError(f"Medium layer cannot be MM method: {medium_spec}")
        if high.basis is None:
            raise ValueError(f"High layer requires a basis set: {high_spec}")
        if medium.basis is None:
            raise ValueError(f"Medium layer requires a basis set: {medium_spec}")

        return cls(
            high_method=high.method,
            high_basis=high.basis,
            medium_method=medium.method,
            medium_basis=medium.basis,
            low_method=low.method,
            low_basis=low.basis,
        )

    def to_gaussian_input(self) -> str:
        """Generate Gaussian-style ONIOM input specification.

        Returns:
            ONIOM specification string for Gaussian input.

        Example:
            >>> scheme = ONIOMScheme("B3LYP", "6-31G*", "UFF")
            >>> scheme.to_gaussian_input()
            'ONIOM(B3LYP/6-31G*:UFF)'
        """
        if self.is_three_layer:
            return (
                f"ONIOM({self.high_method}/{self.high_basis}:"
                f"{self.medium_method}/{self.medium_basis}:{self.low_method})"
            )
        else:
            return f"ONIOM({self.high_method}/{self.high_basis}:{self.low_method})"

    def to_gamess_input(self) -> Dict[str, Any]:
        """Generate GAMESS-style SIMOMM input parameters.

        GAMESS uses the SIMOMM method for QM/MM calculations.

        Returns:
            Dict with GAMESS input parameters.
        """
        params: Dict[str, Any] = {
            "runtyp": "energy",
            "qmmm": True,
        }

        # High layer (QM region)
        params["qmmethod"] = self.high_method
        params["qmbasis"] = self.high_basis

        # Low layer (MM region)
        params["mmmethod"] = self.low_method

        # Embedding type
        if self.electrostatic_embedding:
            params["embedding"] = "electrostatic"
        else:
            params["embedding"] = "mechanical"

        return params

    def to_gamess_simomm_string(self) -> str:
        """Generate GAMESS SIMOMM input section as string.

        Returns:
            GAMESS-formatted input string.
        """
        lines = [" $SIMOMM"]
        lines.append("  QMMM=.TRUE.")
        lines.append(f"  MM={self.low_method}")

        if self.electrostatic_embedding:
            lines.append("  ILINK=1  ! Electrostatic embedding")
        else:
            lines.append("  ILINK=0  ! Mechanical embedding")

        lines.append(" $END")
        return "\n".join(lines)

    def get_layer_atoms(self, layer_name: str) -> "set[int]":
        """Get atoms in a specific layer by name.

        Args:
            layer_name: "high", "medium", or "low".

        Returns:
            Set of atom indices in the layer.
        """
        layer = self.get_layer(layer_name)
        if layer:
            return layer.atom_indices
        return set()

    def set_layer_atoms(self, layer_name: str, atoms: "set[int]") -> None:
        """Set atoms for a specific layer.

        Args:
            layer_name: "high", "medium", or "low".
            atoms: Set of atom indices.
        """
        layer = self.get_layer(layer_name)
        if layer:
            layer.atom_indices = set(atoms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with ONIOM scheme data.
        """
        base_dict = super().to_dict()
        base_dict.update({
            "high_method": self.high_method,
            "high_basis": self.high_basis,
            "medium_method": self.medium_method,
            "medium_basis": self.medium_basis,
            "low_method": self.low_method,
            "low_basis": self.low_basis,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ONIOMScheme":
        """Create from dictionary representation.

        Args:
            data: Dictionary with ONIOM scheme data.

        Returns:
            New ONIOMScheme instance.
        """
        embedding_str = data.get("embedding_type", "electrostatic")
        scheme = cls(
            high_method=data["high_method"],
            high_basis=data["high_basis"],
            low_method=data.get("low_method", "UFF"),
            low_basis=data.get("low_basis"),
            medium_method=data.get("medium_method"),
            medium_basis=data.get("medium_basis"),
            embedding=EmbeddingType(embedding_str),
            name=data.get("name"),
        )

        # Restore layer atoms if present
        if "layers" in data:
            for layer_data in data["layers"]:
                layer = scheme.get_layer(layer_data["name"])
                if layer:
                    layer.atom_indices = set(layer_data.get("atom_indices", []))

        return scheme


def create_oniom_scheme(
    high_method: str,
    high_basis: str,
    high_atoms: "set[int]",
    low_method: str = "UFF",
    low_atoms: Optional["set[int]"] = None,
    medium_method: Optional[str] = None,
    medium_basis: Optional[str] = None,
    medium_atoms: Optional["set[int]"] = None,
    total_atoms: Optional[int] = None,
) -> ONIOMScheme:
    """Convenience function to create a fully configured ONIOM scheme.

    Args:
        high_method: Method for high-level layer.
        high_basis: Basis set for high-level layer.
        high_atoms: Atom indices for high-level layer.
        low_method: Method for low-level layer.
        low_atoms: Atom indices for low layer (computed if None).
        medium_method: Method for medium layer (optional).
        medium_basis: Basis set for medium layer.
        medium_atoms: Atom indices for medium layer.
        total_atoms: Total number of atoms (used to compute low_atoms).

    Returns:
        Configured ONIOMScheme with atoms assigned.
    """
    scheme = ONIOMScheme(
        high_method=high_method,
        high_basis=high_basis,
        low_method=low_method,
        medium_method=medium_method,
        medium_basis=medium_basis,
    )

    # Set high layer atoms
    scheme.set_layer_atoms("high", high_atoms)

    # Set medium layer atoms if 3-layer
    if medium_method and medium_atoms:
        scheme.set_layer_atoms("medium", medium_atoms)

    # Set low layer atoms
    if low_atoms is not None:
        scheme.set_layer_atoms("low", low_atoms)
    elif total_atoms is not None:
        # Compute low atoms as everything not in high/medium
        all_atoms = set(range(total_atoms))
        assigned = high_atoms.copy()
        if medium_atoms:
            assigned.update(medium_atoms)
        low_atoms = all_atoms - assigned
        scheme.set_layer_atoms("low", low_atoms)

    return scheme
