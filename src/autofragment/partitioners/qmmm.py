# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""QM/MM partitioner for multi-level calculations.

This module provides a partitioner specifically designed for QM/MM
calculations, including QM region selection, buffer zone generation,
and link atom placement.

Example:
    >>> from autofragment.partitioners.qmmm import QMMMPartitioner, AtomSelection
    >>> selection = AtomSelection(atom_indices={0, 1, 2, 3, 4})
    >>> partitioner = QMMMPartitioner(qm_selection=selection, buffer_radius=5.0)
    >>> system = ChemicalSystem.from_molecules(molecules)
    >>> result = partitioner.partition(system)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from autofragment.core.bonds import COVALENT_RADII
from autofragment.core.types import ChemicalSystem, FragmentTree
from autofragment.multilevel.layers import (
    ComputationalLayer,
    LayerType,
    LinkAtom,
    MultiLevelScheme,
)
from autofragment.partitioners.topology import (
    BondPolicy,
    SelectionMode,
    TopologyNeighborSelection,
)


@dataclass
class QMMMResult:
    """Result of QM/MM partitioning.

    Attributes:
        qm_atoms: Atom indices in the QM region.
        buffer_atoms: Atom indices in the buffer region.
        mm_atoms: Atom indices in the MM region.
        link_atoms: Link atoms at QM/MM boundaries.
        cut_bonds: List of (qm_idx, mm_idx) tuples for cut bonds.
        qm_charge: Net charge of the QM region.
        qm_multiplicity: Spin multiplicity of the QM region.
    """

    qm_atoms: Set[int]
    buffer_atoms: Set[int]
    mm_atoms: Set[int]
    link_atoms: List[LinkAtom] = field(default_factory=list)
    cut_bonds: List[Tuple[int, int]] = field(default_factory=list)
    qm_charge: int = 0
    qm_multiplicity: int = 1

    @property
    def n_qm_atoms(self) -> int:
        """Number of QM atoms."""
        return len(self.qm_atoms)

    @property
    def n_buffer_atoms(self) -> int:
        """Number of buffer atoms."""
        return len(self.buffer_atoms)

    @property
    def n_mm_atoms(self) -> int:
        """Number of MM atoms."""
        return len(self.mm_atoms)

    @property
    def n_link_atoms(self) -> int:
        """Number of link atoms."""
        return len(self.link_atoms)

    @property
    def total_atoms(self) -> int:
        """Total number of real atoms."""
        return self.n_qm_atoms + self.n_buffer_atoms + self.n_mm_atoms

    def to_multilevel_scheme(
        self,
        qm_method: str = "B3LYP",
        qm_basis: str = "6-31G*",
        mm_method: str = "AMBER",
        name: str = "QM/MM",
    ) -> MultiLevelScheme:
        """Convert to a MultiLevelScheme.

        Args:
            qm_method: Method for QM region.
            qm_basis: Basis set for QM region.
            mm_method: Method for MM region.
            name: Scheme name.

        Returns:
            Configured MultiLevelScheme.
        """
        scheme = MultiLevelScheme(name=name, scheme_type="qmmm")

        # QM layer
        qm_layer = ComputationalLayer(
            name="qm",
            layer_type=LayerType.HIGH,
            method=qm_method,
            basis_set=qm_basis,
            atom_indices=self.qm_atoms.copy(),
            charge=self.qm_charge,
            multiplicity=self.qm_multiplicity,
            link_atoms=self.link_atoms.copy(),
        )
        scheme.add_layer(qm_layer)

        # Buffer layer (treated at MM level but may need special handling)
        if self.buffer_atoms:
            buffer_layer = ComputationalLayer(
                name="buffer",
                layer_type=LayerType.MM,
                method=mm_method,
                atom_indices=self.buffer_atoms.copy(),
            )
            scheme.add_layer(buffer_layer)

        # MM layer
        mm_layer = ComputationalLayer(
            name="mm",
            layer_type=LayerType.MM,
            method=mm_method,
            atom_indices=self.mm_atoms.copy(),
        )
        scheme.add_layer(mm_layer)

        return scheme

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "qm_atoms": sorted(self.qm_atoms),
            "buffer_atoms": sorted(self.buffer_atoms),
            "mm_atoms": sorted(self.mm_atoms),
            "link_atoms": [link.to_dict() for link in self.link_atoms],
            "cut_bonds": list(self.cut_bonds),
            "qm_charge": self.qm_charge,
            "qm_multiplicity": self.qm_multiplicity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QMMMResult":
        """Create from dictionary representation."""
        return cls(
            qm_atoms=set(data["qm_atoms"]),
            buffer_atoms=set(data["buffer_atoms"]),
            mm_atoms=set(data["mm_atoms"]),
            link_atoms=[LinkAtom.from_dict(d) for d in data.get("link_atoms", [])],
            cut_bonds=[(t[0], t[1]) for t in data.get("cut_bonds", [])],
            qm_charge=data.get("qm_charge", 0),
            qm_multiplicity=data.get("qm_multiplicity", 1),
        )


class QMSelection(ABC):
    """Abstract base class for QM region selection strategies.

    Subclasses implement different strategies for selecting which atoms
    should be treated quantum mechanically.
    """

    @abstractmethod
    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Select atoms for the QM region.

        Args:
            coords: (N, 3) array of atomic coordinates.
            elements: List of element symbols for each atom.
            **kwargs: Additional properties (residue, chain, etc.)

        Returns:
            Set of atom indices to include in QM region.
        """
        pass


class AtomSelection(QMSelection):
    """Select QM region by explicit atom indices.

    This is the simplest selection strategy where the user explicitly
    specifies which atoms should be in the QM region.

    Attributes:
        atom_indices: Set of atom indices for QM region.
    """

    def __init__(self, atom_indices: Set[int]):
        """Initialize with explicit atom indices.

        Args:
            atom_indices: Atom indices for QM region.
        """
        self.atom_indices = set(atom_indices)

    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Return the pre-defined atom indices."""
        return self.atom_indices.copy()


class ResidueSelection(QMSelection):
    """Select QM region by residue name or number.

    Useful for protein/biochemical systems where entire residues
    should be in the QM region (e.g., active site residues).

    Attributes:
        residue_names: List of residue names to include.
        residue_numbers: List of residue numbers to include.
    """

    def __init__(
        self,
        residue_names: Optional[List[str]] = None,
        residue_numbers: Optional[List[int]] = None,
    ):
        """Initialize with residue selection criteria.

        Args:
            residue_names: Residue names to include (e.g., ["HIS", "CYS"]).
            residue_numbers: Residue numbers to include (e.g., [64, 68]).
        """
        self.residue_names = set(residue_names or [])
        self.residue_numbers = set(residue_numbers or [])

    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Select atoms by residue."""
        residue_names = kwargs.get("residue_names", [])
        residue_numbers = kwargs.get("residue_numbers", [])

        selected: Set[int] = set()

        for i in range(len(coords)):
            # Check by name
            if residue_names and i < len(residue_names):
                if residue_names[i] in self.residue_names:
                    selected.add(i)
                    continue

            # Check by number
            if residue_numbers and i < len(residue_numbers):
                if residue_numbers[i] in self.residue_numbers:
                    selected.add(i)

        return selected


class DistanceSelection(QMSelection):
    """Select QM region by distance from a center point.

    Selects all atoms within a specified radius of a center point.
    Useful for defining spherical QM regions around active sites.

    Attributes:
        center: Center point coordinates (3,).
        radius: Selection radius in Angstroms.
    """

    def __init__(self, center: np.ndarray, radius: float):
        """Initialize with center and radius.

        Args:
            center: Center point (3,) array.
            radius: Radius in Angstroms.
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius

    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Select atoms within radius of center."""
        selected: Set[int] = set()

        for i in range(len(coords)):
            dist = np.linalg.norm(coords[i] - self.center)
            if dist <= self.radius:
                selected.add(i)

        return selected


class CombinedSelection(QMSelection):
    """Combine multiple selection strategies.

    Can perform union (OR) or intersection (AND) of multiple
    selection strategies.

    Attributes:
        selections: List of selection strategies.
        mode: "union" for OR, "intersection" for AND.
    """

    def __init__(
        self,
        selections: List[QMSelection],
        mode: str = "union",
    ):
        """Initialize with multiple selections.

        Args:
            selections: List of QMSelection objects.
            mode: "union" (any match) or "intersection" (all match).
        """
        self.selections = selections
        self.mode = mode

    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Combine selections according to mode."""
        if not self.selections:
            return set()

        results = [s.select(coords, elements, **kwargs) for s in self.selections]

        if self.mode == "union":
            combined: Set[int] = set()
            for r in results:
                combined.update(r)
            return combined
        else:  # intersection
            combined = results[0].copy()
            for r in results[1:]:
                combined &= r
            return combined


class TopologySelection(QMSelection):
    """Select QM region by topology or nearest-neighbor neighborhood shells.

    Supports graph-hop expansion from seed atoms (chemically preferred)
    and Euclidean layered nearest-neighbor expansion.
    """

    def __init__(
        self,
        seed_atoms: Set[int],
        mode: SelectionMode = "graph",
        hops: int = 1,
        layers: int = 1,
        k_per_layer: int = 1,
        expand_residues: bool = False,
        bond_policy: BondPolicy = "infer",
    ):
        """Initialize a new TopologySelection instance."""
        self.selector = TopologyNeighborSelection(
            seed_atoms=set(seed_atoms),
            mode=mode,
            hops=hops,
            layers=layers,
            k_per_layer=k_per_layer,
            expand_residues=expand_residues,
            bond_policy=bond_policy,
        )

    def select(
        self,
        coords: np.ndarray,
        elements: List[str],
        **kwargs: Any,
    ) -> Set[int]:
        """Select atoms using configured topology neighborhood strategy."""
        result = self.selector.select(
            coords,
            elements,
            bonds=kwargs.get("bonds"),
            residue_names=kwargs.get("residue_names"),
            residue_numbers=kwargs.get("residue_numbers"),
            chains=kwargs.get("chains"),
        )
        return result.selected_atoms


class QMMMPartitioner:
    """Partitioner for QM/MM calculations.

    Divides a molecular system into QM, buffer, and MM regions,
    identifying QM/MM boundaries and placing link atoms.

    Attributes:
        qm_selection: Strategy for selecting QM atoms.
        buffer_radius: Radius for buffer region around QM (Angstroms).
        link_scheme: Link atom placement scheme ("hydrogen", "scaled").
    """

    def __init__(
        self,
        qm_selection: QMSelection,
        buffer_radius: float = 5.0,
        link_scheme: str = "hydrogen",
        expand_to_groups: bool = True,
    ):
        """Initialize the QM/MM partitioner.

        Args:
            qm_selection: Selection strategy for QM region.
            buffer_radius: Buffer radius in Angstroms (default 5.0).
            link_scheme: Link atom scheme ("hydrogen" or "scaled").
            expand_to_groups: Whether to expand to complete groups.
        """
        self.qm_selection = qm_selection
        self.buffer_radius = buffer_radius
        self.link_scheme = link_scheme
        self.expand_to_groups = expand_to_groups

    def partition(
        self,
        system: ChemicalSystem,
        source_file: str | None = None,
        bonds: Optional[List[Tuple[int, int]]] = None,
        qm_charge: int = 0,
        qm_multiplicity: int = 1,
        **kwargs: Any,
    ) -> QMMMResult:
        """Partition a chemical system into QM/MM regions.

        Args:
            system: ChemicalSystem to partition.
            source_file: Optional source file path.
            bonds: Optional list of (i, j) bonded pairs.
            qm_charge: Charge of the QM region.
            qm_multiplicity: Multiplicity of the QM region.
            **kwargs: Additional properties for selection.

        Returns:
            QMMMResult with all region assignments.
        """
        # Flatten system to coordinates and elements
        all_coords = []
        all_elements = []
        atoms = system.atoms
        if bonds is None and system.bonds:
            bonds = [
                (bond["atom1"], bond["atom2"])
                for bond in system.bonds
                if "atom1" in bond and "atom2" in bond
            ]

        for atom in atoms:
            all_coords.append(atom.coords)
            all_elements.append(atom.symbol)

        coords = np.array(all_coords)
        n_atoms = len(coords)

        # 1. Find bonds if not provided
        if bonds is None:
            bonds = self._detect_bonds(coords, all_elements)

        # 2. Select QM region
        qm_atoms = self.qm_selection.select(
            coords,
            all_elements,
            bonds=bonds,
            **kwargs,
        )

        # 3. Find cut bonds (QM-MM boundaries)
        cut_bonds = self._find_cut_bonds(qm_atoms, bonds)

        # 4. Generate buffer region
        buffer_atoms = self._generate_buffer(coords, qm_atoms)

        # 5. Position link atoms
        link_atoms = self._position_link_atoms(coords, all_elements, cut_bonds)

        # 6. MM region is everything else
        mm_atoms = set(range(n_atoms)) - qm_atoms - buffer_atoms

        return QMMMResult(
            qm_atoms=qm_atoms,
            buffer_atoms=buffer_atoms,
            mm_atoms=mm_atoms,
            link_atoms=link_atoms,
            cut_bonds=cut_bonds,
            qm_charge=qm_charge,
            qm_multiplicity=qm_multiplicity,
        )

    def _detect_bonds(
        self,
        coords: np.ndarray,
        elements: List[str],
        tolerance: float = 0.4,
    ) -> List[Tuple[int, int]]:
        """Detect bonds based on covalent radii.

        Args:
            coords: Atomic coordinates (N, 3).
            elements: Element symbols.
            tolerance: Bond tolerance in Angstroms.

        Returns:
            List of (i, j) bonded pairs.
        """
        n_atoms = len(coords)
        bonds: List[Tuple[int, int]] = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                r_i = COVALENT_RADII.get(elements[i], 1.5)
                r_j = COVALENT_RADII.get(elements[j], 1.5)
                max_bond = r_i + r_j + tolerance

                if dist <= max_bond:
                    bonds.append((i, j))

        return bonds

    def _find_cut_bonds(
        self,
        qm_atoms: Set[int],
        bonds: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Find bonds that cross the QM/MM boundary.

        Args:
            qm_atoms: Set of QM atom indices.
            bonds: List of bonded pairs.

        Returns:
            List of (qm_idx, mm_idx) tuples for cut bonds.
        """
        cut_bonds: List[Tuple[int, int]] = []

        for i, j in bonds:
            i_in_qm = i in qm_atoms
            j_in_qm = j in qm_atoms

            if i_in_qm and not j_in_qm:
                cut_bonds.append((i, j))  # i is QM, j is MM
            elif j_in_qm and not i_in_qm:
                cut_bonds.append((j, i))  # j is QM, i is MM

        return cut_bonds

    def _generate_buffer(
        self,
        coords: np.ndarray,
        qm_atoms: Set[int],
    ) -> Set[int]:
        """Generate buffer region around QM atoms.

        Args:
            coords: Atomic coordinates.
            qm_atoms: QM atom indices.

        Returns:
            Set of buffer atom indices.
        """
        if self.buffer_radius <= 0:
            return set()

        buffer_atoms: Set[int] = set()
        n_atoms = len(coords)

        for i in range(n_atoms):
            if i in qm_atoms:
                continue

            # Check distance to any QM atom
            for qm_idx in qm_atoms:
                dist = np.linalg.norm(coords[i] - coords[qm_idx])
                if dist <= self.buffer_radius:
                    buffer_atoms.add(i)
                    break

        return buffer_atoms

    def _position_link_atoms(
        self,
        coords: np.ndarray,
        elements: List[str],
        cut_bonds: List[Tuple[int, int]],
    ) -> List[LinkAtom]:
        """Position link atoms along cut bonds.

        Args:
            coords: Atomic coordinates.
            elements: Element symbols.
            cut_bonds: List of (qm_idx, mm_idx) tuples.

        Returns:
            List of positioned LinkAtom objects.
        """
        link_atoms: List[LinkAtom] = []

        for qm_idx, mm_idx in cut_bonds:
            # Get positions
            qm_pos = coords[qm_idx]
            mm_pos = coords[mm_idx]

            # Compute bond vector
            bond_vec = mm_pos - qm_pos
            bond_length = float(np.linalg.norm(bond_vec))

            # Determine link atom position based on scheme
            if self.link_scheme == "hydrogen":
                # Standard hydrogen link atom
                # C-H bond length is ~1.09 Angstrom
                link_distance = 1.09
                scale_factor: float = link_distance / bond_length
            else:  # scaled
                # Use scaled position along the bond
                scale_factor = 0.723  # Standard for C-C to C-H

            link_pos = qm_pos + scale_factor * bond_vec

            link_atom = LinkAtom(
                qm_atom_index=qm_idx,
                mm_atom_index=mm_idx,
                element="H",
                position=link_pos,
                scale_factor=scale_factor,
            )
            link_atoms.append(link_atom)

        return link_atoms

    def partition_to_fragment_tree(
        self,
        system: ChemicalSystem,
        source_file: str | None = None,
        **kwargs: Any,
    ) -> FragmentTree:
        """Partition and return as FragmentTree for compatibility.

        Args:
            system: ChemicalSystem to partition.
            source_file: Optional source file.
            **kwargs: Additional parameters.

        Returns:
            FragmentTree with QM and MM fragments.
        """
        from autofragment.core.types import Fragment

        result = self.partition(system, source_file, **kwargs)

        # Flatten atoms
        all_symbols = []
        all_coords = []
        for atom in system.atoms:
            all_symbols.append(atom.symbol)
            all_coords.extend(atom.coords.tolist())

        # Create QM fragment
        qm_symbols = [all_symbols[i] for i in sorted(result.qm_atoms)]
        qm_geom = []
        for i in sorted(result.qm_atoms):
            idx = i * 3
            qm_geom.extend(all_coords[idx:idx+3])

        qm_fragment = Fragment(
            id="QM",
            symbols=qm_symbols,
            geometry=qm_geom,
            molecular_charge=result.qm_charge,
            molecular_multiplicity=result.qm_multiplicity,
        )

        # Create MM fragment
        mm_symbols = [all_symbols[i] for i in sorted(result.mm_atoms | result.buffer_atoms)]
        mm_geom = []
        for i in sorted(result.mm_atoms | result.buffer_atoms):
            idx = i * 3
            mm_geom.extend(all_coords[idx:idx+3])

        mm_fragment = Fragment(
            id="MM",
            symbols=mm_symbols,
            geometry=mm_geom,
            molecular_charge=0,
            molecular_multiplicity=1,
        )

        return FragmentTree(
            fragments=[qm_fragment, mm_fragment],
            partitioning={"partition_type": "qmmm"},
        )
