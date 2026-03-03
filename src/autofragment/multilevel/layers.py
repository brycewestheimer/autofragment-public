# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Computational layer definitions for multi-level methods.

This module defines data structures for representing multi-level
computational schemes like ONIOM, where different regions of a
molecular system are treated at different levels of theory.

Example:
    >>> from autofragment.multilevel import LayerType, ComputationalLayer
    >>> qm_layer = ComputationalLayer(
    ...     name="active_site",
    ...     layer_type=LayerType.HIGH,
    ...     method="B3LYP",
    ...     basis_set="6-31G*",
    ...     atom_indices={0, 1, 2, 3, 4}
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:
    pass


class LayerType(Enum):
    """Type of computational layer in a multi-level scheme.

    Attributes:
        HIGH: Highest level QM treatment (e.g., CCSD(T), MP2)
        MEDIUM: Medium level QM treatment (e.g., DFT)
        LOW: Low level QM treatment (e.g., HF, semi-empirical)
        MM: Molecular mechanics treatment
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MM = "mm"


@dataclass
class LinkAtom:
    """A link atom placed at a covalent boundary between layers.

    Link atoms are used to cap dangling bonds when a covalent bond
    crosses the boundary between two computational layers.

    Attributes:
        qm_atom_index: Index of the QM atom (the one kept in higher layer)
        mm_atom_index: Index of the MM atom (the one in lower layer)
        element: Element symbol of the link atom (typically 'H')
        position: 3D coordinates of the link atom
        scale_factor: Distance scaling factor (link distance = scale * bond length)
    """

    qm_atom_index: int
    mm_atom_index: int
    element: str = "H"
    position: Optional[np.ndarray] = None
    scale_factor: float = 0.723  # Default for C-H replacing C-C

    def __post_init__(self) -> None:
        """Validate and convert position array."""
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=np.float64)
            if self.position.shape != (3,):
                raise ValueError(
                    f"LinkAtom position must be a 3D vector, got shape {self.position.shape}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with link atom data.
        """
        return {
            "qm_atom_index": self.qm_atom_index,
            "mm_atom_index": self.mm_atom_index,
            "element": self.element,
            "position": self.position.tolist() if self.position is not None else None,
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkAtom":
        """Create from dictionary representation.

        Args:
            data: Dictionary with link atom data.

        Returns:
            New LinkAtom instance.
        """
        position = data.get("position")
        if position is not None:
            position = np.array(position, dtype=np.float64)

        return cls(
            qm_atom_index=data["qm_atom_index"],
            mm_atom_index=data["mm_atom_index"],
            element=data.get("element", "H"),
            position=position,
            scale_factor=data.get("scale_factor", 0.723),
        )


@dataclass
class ComputationalLayer:
    """A single computational layer in a multi-level scheme.

    A computational layer represents a region of a molecular system
    that will be treated at a specific level of theory. In ONIOM-style
    calculations, multiple layers are combined to achieve accurate
    results at reduced computational cost.

    Attributes:
        name: Human-readable layer name (e.g., "active_site", "protein")
        layer_type: HIGH, MEDIUM, LOW, or MM
        method: Computational method (e.g., "B3LYP", "CCSD(T)", "AMBER")
        basis_set: Basis set for QM methods (None for MM)
        atom_indices: Indices of atoms belonging to this layer
        charge: Total charge of this layer
        multiplicity: Spin multiplicity of this layer
        frozen_atoms: Indices of atoms frozen during optimization
        link_atoms: Link atoms at boundaries with other layers

    Example:
        >>> layer = ComputationalLayer(
        ...     name="qm_region",
        ...     layer_type=LayerType.HIGH,
        ...     method="B3LYP",
        ...     basis_set="6-31G*",
        ...     atom_indices={0, 1, 2},
        ...     charge=0,
        ...     multiplicity=1
        ... )
        >>> layer.n_atoms
        3
    """

    name: str
    layer_type: LayerType
    method: str
    basis_set: Optional[str] = None
    atom_indices: Set[int] = field(default_factory=set)
    charge: int = 0
    multiplicity: int = 1
    frozen_atoms: Set[int] = field(default_factory=set)
    link_atoms: List[LinkAtom] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate layer configuration."""
        # QM layers require a basis set
        if self.layer_type != LayerType.MM and self.basis_set is None:
            raise ValueError(
                f"QM layer '{self.name}' requires a basis set. "
                f"Only MM layers can have basis_set=None."
            )

        # Ensure atom_indices is a set
        if not isinstance(self.atom_indices, set):
            self.atom_indices = set(self.atom_indices)

        # Ensure frozen_atoms is a set
        if not isinstance(self.frozen_atoms, set):
            self.frozen_atoms = set(self.frozen_atoms)

        # Validate frozen atoms are in layer
        if self.frozen_atoms and not self.frozen_atoms.issubset(self.atom_indices):
            invalid = self.frozen_atoms - self.atom_indices
            raise ValueError(
                f"Frozen atoms {invalid} are not in layer '{self.name}'"
            )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this layer (excluding link atoms)."""
        return len(self.atom_indices)

    @property
    def total_atoms(self) -> int:
        """Total number of atoms including link atoms."""
        return len(self.atom_indices) + len(self.link_atoms)

    @property
    def is_qm(self) -> bool:
        """Check if this is a QM layer (not MM)."""
        return self.layer_type != LayerType.MM

    def add_atoms(self, indices: Iterable[int]) -> None:
        """Add atoms to this layer.

        Args:
            indices: Atom indices to add.
        """
        self.atom_indices.update(indices)

    def remove_atoms(self, indices: Iterable[int]) -> None:
        """Remove atoms from this layer.

        Args:
            indices: Atom indices to remove.
        """
        self.atom_indices -= set(indices)
        # Also remove from frozen atoms if present
        self.frozen_atoms -= set(indices)

    def add_link_atom(
        self,
        qm_atom_index: int,
        mm_atom_index: int,
        element: str = "H",
        position: Optional[np.ndarray] = None,
        scale_factor: float = 0.723,
    ) -> None:
        """Add a link atom at a boundary.

        Args:
            qm_atom_index: Index of the QM atom (in this layer).
            mm_atom_index: Index of the MM atom (in lower layer).
            element: Element symbol of the link atom.
            position: 3D position of the link atom.
            scale_factor: Distance scaling factor.
        """
        link = LinkAtom(
            qm_atom_index=qm_atom_index,
            mm_atom_index=mm_atom_index,
            element=element,
            position=position,
            scale_factor=scale_factor,
        )
        self.link_atoms.append(link)

    def get_boundary_atoms(self) -> Set[int]:
        """Get QM atoms at layer boundaries (those with link atoms).

        Returns:
            Set of atom indices that are at boundaries.
        """
        return {link.qm_atom_index for link in self.link_atoms}

    def freeze_atoms(self, indices: Iterable[int]) -> None:
        """Mark atoms as frozen during optimization.

        Args:
            indices: Atom indices to freeze.

        Raises:
            ValueError: If any index is not in this layer.
        """
        indices_set = set(indices)
        if not indices_set.issubset(self.atom_indices):
            invalid = indices_set - self.atom_indices
            raise ValueError(
                f"Cannot freeze atoms {invalid} - not in layer '{self.name}'"
            )
        self.frozen_atoms.update(indices_set)

    def unfreeze_atoms(self, indices: Optional[Iterable[int]] = None) -> None:
        """Unfreeze atoms.

        Args:
            indices: Atom indices to unfreeze. If None, unfreeze all.
        """
        if indices is None:
            self.frozen_atoms.clear()
        else:
            self.frozen_atoms -= set(indices)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with layer data.
        """
        return {
            "name": self.name,
            "layer_type": self.layer_type.value,
            "method": self.method,
            "basis_set": self.basis_set,
            "atom_indices": sorted(self.atom_indices),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "frozen_atoms": sorted(self.frozen_atoms),
            "link_atoms": [link.to_dict() for link in self.link_atoms],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputationalLayer":
        """Create from dictionary representation.

        Args:
            data: Dictionary with layer data.

        Returns:
            New ComputationalLayer instance.
        """
        return cls(
            name=data["name"],
            layer_type=LayerType(data["layer_type"]),
            method=data["method"],
            basis_set=data.get("basis_set"),
            atom_indices=set(data.get("atom_indices", [])),
            charge=data.get("charge", 0),
            multiplicity=data.get("multiplicity", 1),
            frozen_atoms=set(data.get("frozen_atoms", [])),
            link_atoms=[
                LinkAtom.from_dict(link) for link in data.get("link_atoms", [])
            ],
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ComputationalLayer(name='{self.name}', type={self.layer_type.value}, "
            f"method='{self.method}', n_atoms={self.n_atoms})"
        )


class EmbeddingType(Enum):
    """Type of embedding for QM/MM calculations.

    Attributes:
        MECHANICAL: MM region affects QM only through geometry (van der Waals)
        ELECTROSTATIC: MM point charges are included in QM Hamiltonian
        POLARIZABLE: MM region has polarizable dipoles (advanced)
    """

    MECHANICAL = "mechanical"
    ELECTROSTATIC = "electrostatic"
    POLARIZABLE = "polarizable"


@dataclass
class MultiLevelScheme:
    """Container for multiple computational layers.

    A MultiLevelScheme organizes a hierarchy of computational layers
    for multi-level methods like ONIOM. Layers are ordered from highest
    to lowest level of theory.

    Attributes:
        name: Scheme name (e.g., "ONIOM(B3LYP:AMBER)")
        layers: List of computational layers, ordered high to low
        description: Optional description of the scheme
        scheme_type: Type of multi-level method ("oniom", "qmmm", "subtractive")
        embedding_type: Type of embedding for QM/MM interface
        electrostatic_embedding: Whether to use electrostatic embedding
        mechanical_embedding: Whether to use mechanical embedding

    Example:
        >>> scheme = MultiLevelScheme(name="two_layer")
        >>> scheme.add_layer(qm_layer)
        >>> scheme.add_layer(mm_layer)
        >>> scheme.validate()
    """

    name: str
    layers: List[ComputationalLayer] = field(default_factory=list)
    description: str = ""

    # Scheme configuration
    scheme_type: str = "oniom"  # "oniom", "qmmm", "subtractive"
    embedding_type: EmbeddingType = EmbeddingType.ELECTROSTATIC

    # Convenience embedding flags (derived from embedding_type)
    electrostatic_embedding: bool = True
    mechanical_embedding: bool = False

    def __post_init__(self) -> None:
        """Sync embedding flags with embedding type."""
        if self.embedding_type == EmbeddingType.ELECTROSTATIC:
            self.electrostatic_embedding = True
            self.mechanical_embedding = False
        elif self.embedding_type == EmbeddingType.MECHANICAL:
            self.electrostatic_embedding = False
            self.mechanical_embedding = True
        elif self.embedding_type == EmbeddingType.POLARIZABLE:
            self.electrostatic_embedding = True  # Polarizable includes electrostatic
            self.mechanical_embedding = False

    @property
    def n_layers(self) -> int:
        """Number of layers in this scheme."""
        return len(self.layers)

    @property
    def total_atoms(self) -> int:
        """Total number of atoms across all layers."""
        return sum(layer.n_atoms for layer in self.layers)

    def get_high_layer(self) -> Optional[ComputationalLayer]:
        """Get the highest-level (first HIGH type) layer.

        Returns:
            The first HIGH layer if found, None otherwise.
        """
        for layer in self.layers:
            if layer.layer_type == LayerType.HIGH:
                return layer
        return None

    def add_layer(self, layer: ComputationalLayer) -> None:
        """Add a layer to the scheme.

        Args:
            layer: Computational layer to add.
        """
        self.layers.append(layer)

    def get_layer(self, name: str) -> Optional[ComputationalLayer]:
        """Get a layer by name.

        Args:
            name: Layer name to find.

        Returns:
            The layer if found, None otherwise.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def get_layer_by_type(self, layer_type: LayerType) -> List[ComputationalLayer]:
        """Get all layers of a given type.

        Args:
            layer_type: Type of layer to find.

        Returns:
            List of matching layers.
        """
        return [layer for layer in self.layers if layer.layer_type == layer_type]

    def get_atom_layer(self, atom_index: int) -> Optional[ComputationalLayer]:
        """Find which layer contains a given atom.

        Args:
            atom_index: Index of the atom to find.

        Returns:
            The layer containing the atom, or None.
        """
        for layer in self.layers:
            if atom_index in layer.atom_indices:
                return layer
        return None

    def get_all_atoms(self) -> Set[int]:
        """Get all atom indices across all layers.

        Returns:
            Set of all atom indices.
        """
        all_atoms: Set[int] = set()
        for layer in self.layers:
            all_atoms.update(layer.atom_indices)
        return all_atoms

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the multi-level scheme.

        Checks for:
        - No overlapping atoms between layers
        - Each layer has atoms
        - Layer types are ordered correctly

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []

        if not self.layers:
            errors.append("Scheme has no layers")
            return False, errors

        # Check for overlapping atoms
        seen_atoms: Set[int] = set()
        for layer in self.layers:
            overlap = seen_atoms & layer.atom_indices
            if overlap:
                errors.append(
                    f"Atoms {overlap} appear in multiple layers "
                    f"(found in '{layer.name}')"
                )
            seen_atoms.update(layer.atom_indices)

        # Check each layer has atoms
        for layer in self.layers:
            if not layer.atom_indices:
                errors.append(f"Layer '{layer.name}' has no atoms")

        return len(errors) == 0, errors

    def get_oniom_string(self) -> str:
        """Generate ONIOM-style method string.

        Returns:
            String like "ONIOM(B3LYP/6-31G*:AMBER)"
        """
        parts = []
        for layer in self.layers:
            if layer.is_qm:
                parts.append(f"{layer.method}/{layer.basis_set}")
            else:
                parts.append(layer.method)
        return f"ONIOM({':'.join(parts)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with scheme data.
        """
        return {
            "name": self.name,
            "description": self.description,
            "scheme_type": self.scheme_type,
            "embedding_type": self.embedding_type.value,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiLevelScheme":
        """Create from dictionary representation.

        Args:
            data: Dictionary with scheme data.

        Returns:
            New MultiLevelScheme instance.
        """
        embedding_type_str = data.get("embedding_type", "electrostatic")
        scheme = cls(
            name=data["name"],
            description=data.get("description", ""),
            scheme_type=data.get("scheme_type", "oniom"),
            embedding_type=EmbeddingType(embedding_type_str),
        )
        for layer_data in data.get("layers", []):
            scheme.add_layer(ComputationalLayer.from_dict(layer_data))
        return scheme

    def __repr__(self) -> str:
        """Return string representation."""
        layer_info = ", ".join(
            f"{layer.name}({layer.n_atoms})" for layer in self.layers
        )
        return f"MultiLevelScheme(name='{self.name}', layers=[{layer_info}])"
