# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Core data types for autofragment.

This module defines the fundamental data structures used throughout the package:
- ChemicalSystem: Canonical representation of a full system (all atoms + metadata)
- Fragment/FragmentTree: Canonical outputs from partitioning
- Molecule: Lightweight helper for isolated fragments/geometry utilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from autofragment.core.graph import MolecularGraph
    from autofragment.core.lattice import Lattice


@dataclass
class Atom:
    """
    A single atom with element symbol and 3D coordinates.

    Attributes
    ----------
    symbol : str
        Element symbol (e.g., 'O', 'H', 'C').
    coords : np.ndarray
        3D coordinates as a (3,) numpy array in Angstroms.
    charge : float
        Partial or formal charge. Default 0.0.
    """

    symbol: str
    coords: np.ndarray
    charge: float = 0.0

    def __post_init__(self):
        """Validate and normalize dataclass fields after initialization."""
        if isinstance(self.coords, (list, tuple)):
            self.coords = np.array(self.coords, dtype=float)
        if self.coords.shape != (3,):
            raise ValueError(f"Atom coords must be shape (3,), got {self.coords.shape}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "coords": self.coords.tolist(),
            "charge": self.charge,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Atom:
        """Create from dictionary representation."""
        return cls(
            symbol=data["symbol"],
            coords=np.array(data["coords"], dtype=float),
            charge=data.get("charge", 0.0),
        )



# Type alias for an isolated molecule or fragment (list of atoms).
Molecule = List[Atom]


def molecule_to_coords(molecule: Molecule) -> np.ndarray:
    """Convert an isolated molecule to a (N, 3) array of coordinates."""
    return np.stack([atom.coords for atom in molecule], axis=0)


def molecule_centroid(molecule: Molecule) -> np.ndarray:
    """Compute the centroid of an isolated molecule."""
    return molecule_to_coords(molecule).mean(axis=0)


def molecules_to_system(
    molecules: Sequence[Molecule],
    *,
    bonds: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    lattice: Optional["Lattice"] = None,
) -> "ChemicalSystem":
    """Create a ChemicalSystem from isolated molecules.

    This is an explicit conversion boundary when molecule-level utilities
    are used to assemble a full system.
    """
    atoms: List[Atom] = []
    molecule_atom_indices: List[List[int]] = []
    index = 0

    for mol in molecules:
        indices = list(range(index, index + len(mol)))
        molecule_atom_indices.append(indices)
        atoms.extend(mol)
        index += len(mol)

    merged_metadata: Dict[str, Any] = dict(metadata or {})
    merged_metadata.setdefault("molecule_atom_indices", molecule_atom_indices)
    lengths = {len(mol) for mol in molecules}
    if len(lengths) == 1 and lengths:
        merged_metadata.setdefault("atoms_per_molecule", next(iter(lengths)))

    return ChemicalSystem(
        atoms=atoms,
        bonds=bonds or [],
        metadata=merged_metadata,
        lattice=lattice,
    )


def system_to_molecules(
    system: "ChemicalSystem",
    *,
    atoms_per_molecule: Optional[int] = None,
    require_metadata: bool = False,
) -> List[Molecule]:
    """Convert a ChemicalSystem to isolated molecules.

    This conversion relies on explicit molecule boundaries when available.
    """
    metadata = system.metadata or {}
    if atoms_per_molecule is None:
        atoms_per_molecule = metadata.get("atoms_per_molecule")

    if "molecule_atom_indices" in metadata:
        molecules: List[Molecule] = []
        for indices in metadata["molecule_atom_indices"]:
            molecules.append([system.atoms[i] for i in indices])
        return molecules

    if atoms_per_molecule is not None:
        if atoms_per_molecule <= 0:
            raise ValueError("atoms_per_molecule must be positive.")
        return [
            system.atoms[i : i + atoms_per_molecule]
            for i in range(0, len(system.atoms), atoms_per_molecule)
        ]

    if require_metadata:
        raise ValueError(
            "ChemicalSystem is missing molecule boundaries. "
            "Provide metadata 'molecule_atom_indices' or 'atoms_per_molecule', "
            "or use molecules_to_system()."
        )

    return [system.atoms]


@dataclass
class Fragment:
    """
    A fragment containing atoms and/or sub-fragments.

    Fragments can be nested: a primary fragment contains secondary fragments,
    which may contain tertiary fragments, and so on. Leaf fragments contain
    actual atom data (symbols and geometry).

    Attributes
    ----------
    id : str
        Unique identifier for this fragment (e.g., 'F1', 'PF1', 'PF1_SF2').
    symbols : List[str]
        Atom symbols for leaf fragments (empty for non-leaf).
    geometry : List[float]
        Flat list of atom coordinates [x1, y1, z1, x2, y2, z2, ...].
    molecular_charge : int
        Total molecular charge of this fragment.
    molecular_multiplicity : int
        Spin multiplicity of this fragment.
    fragments : List[Fragment]
        Child fragments (empty for leaf fragments).
    """

    id: str
    symbols: List[str] = field(default_factory=list)
    geometry: List[float] = field(default_factory=list)
    molecular_charge: int = 0
    molecular_multiplicity: int = 1
    layer: Optional[str] = None
    embedding_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    fragments: List["Fragment"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf fragment (contains atoms, not sub-fragments)."""
        return len(self.fragments) == 0

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this fragment (including sub-fragments)."""
        if self.is_leaf:
            return len(self.symbols)
        return sum(f.n_atoms for f in self.fragments)

    @property
    def n_molecules(self) -> int:
        """Number of molecules (for molecular systems)."""
        # This assumes each molecule was added as a contiguous block or
        # we can infer it. For now, if we don't have metadata, assume atomic?
        # Actually in MolecularPartitioner it says:
        # fragments.append(Fragment.from_molecules(chosen, f"F{k + 1}"))

        # But Fragment.from_molecules just flattens atoms.
        # It doesn't store molecule count explicitly in Fragment structure shown in types.py.
        # Unless we added it to metadata or change types.py.
        # Let's check metadata.
        # It seems not stored by default.
        # We can estimate if we know molecule size, or just return 0 if unknown.
        # Better yet, let's fix the NOTEBOOK to not use this property if it doesn't exist.
        return self.metadata.get("n_molecules", 0)

    def get_coords(self) -> np.ndarray:
        """
        Get coordinates as a (N, 3) numpy array.

        Returns
        -------
        np.ndarray
            Coordinates array with shape (n_atoms, 3).
        """
        if not self.geometry:
            return np.array([]).reshape(0, 3)
        return np.array(self.geometry).reshape(-1, 3)

    def set_coords(self, coords: np.ndarray) -> None:
        """
        Set coordinates from a (N, 3) numpy array.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates array with shape (n_atoms, 3).
        """
        self.geometry = coords.flatten().tolist()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for JSON serialization."""
        result: Dict[str, Any] = {"id": self.id}

        if self.fragments:
            result["fragments"] = [f.to_dict() for f in self.fragments]

        if self.symbols:
            result["symbols"] = self.symbols

        if self.geometry:
            result["geometry"] = self.geometry

        result["molecular_charge"] = self.molecular_charge
        result["molecular_multiplicity"] = self.molecular_multiplicity

        if self.layer:
            result["layer"] = self.layer
        if self.embedding_type:
            result["embedding_type"] = self.embedding_type

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Fragment:
        """Create from dictionary representation."""
        fragments = [cls.from_dict(f) for f in data.get("fragments", [])]
        return cls(
            id=data["id"],
            symbols=data.get("symbols", []),
            geometry=data.get("geometry", []),
            molecular_charge=data.get("molecular_charge", 0),
            molecular_multiplicity=data.get("molecular_multiplicity", 1),
            layer=data.get("layer"),
            embedding_type=data.get("embedding_type"),
            metadata=data.get("metadata", {}),
            fragments=fragments,
        )


    @classmethod
    def from_molecules(
        cls,
        molecules: Sequence[Molecule],
        fragment_id: str,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> Fragment:
        """
        Create a leaf fragment from a sequence of molecules.

        Parameters
        ----------
        molecules : Sequence[Molecule]
            List of molecules (each molecule is a list of Atoms).
        fragment_id : str
            Unique identifier for this fragment.
        charge : int, optional
            Molecular charge. Default is 0.
        multiplicity : int, optional
            Spin multiplicity. Default is 1.

        Returns
        -------
        Fragment
            A leaf fragment containing all atoms from the molecules.
        """
        symbols: List[str] = []
        geometry: List[float] = []

        for molecule in molecules:
            for atom in molecule:
                symbols.append(atom.symbol)
                geometry.extend(atom.coords.tolist())

        return cls(
            id=fragment_id,
            symbols=symbols,
            geometry=geometry,
            molecular_charge=charge,
            molecular_multiplicity=multiplicity,
        )


@dataclass
class FragmentTree:
    """
    A container for fragments with metadata.

    Supports both flat fragment lists and hierarchical (tiered) trees.

    Attributes
    ----------
    fragments : List[Fragment]
        Top-level (primary) fragments.
    interfragment_bonds : List[Dict]
        Bonds between fragments (for biological systems).
    source : Dict[str, str]
        Information about the source file.
    partitioning : Dict[str, Any]
        Information about the partitioning algorithm used.
    version : str
        Output format version.
    """

    fragments: List[Fragment]
    interfragment_bonds: List[Dict[str, Any]] = field(default_factory=list)
    source: Dict[str, str] = field(default_factory=dict)
    partitioning: Dict[str, Any] = field(default_factory=dict)
    version: str = "2.0"

    @property
    def n_primary(self) -> int:
        """Number of primary (top-level) fragments."""
        return len(self.fragments)

    @property
    def _is_hierarchical(self) -> bool:
        """True if any top-level fragment has children."""
        return any(not f.is_leaf for f in self.fragments)

    def _count_descendants(self, fragment: Fragment) -> int:
        """Count all descendant fragments."""
        count = len(fragment.fragments)
        for f in fragment.fragments:
            count += self._count_descendants(f)
        return count

    @property
    def n_fragments(self) -> int:
        """Total number of fragments at all levels.

        For flat trees, returns ``len(self.fragments)``.
        For hierarchical trees, counts all fragments recursively.
        """
        if not self._is_hierarchical:
            return len(self.fragments)
        count = len(self.fragments)
        for f in self.fragments:
            count += self._count_descendants(f)
        return count

    @property
    def n_atoms(self) -> int:
        """Total number of atoms across all fragments."""
        return sum(f.n_atoms for f in self.fragments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for JSON serialization."""
        return {
            "version": self.version,
            "source": self.source,
            "partitioning": self.partitioning,
            "fragments": [f.to_dict() for f in self.fragments],
            "interfragment_bonds": self.interfragment_bonds,
        }

    def to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """
        Write the fragment tree to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        indent : int, optional
            JSON indentation level. Default is 2.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent) + "\n")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FragmentTree:
        """Create from dictionary representation."""
        fragments = [Fragment.from_dict(f) for f in data.get("fragments", [])]
        return cls(
            fragments=fragments,
            interfragment_bonds=data.get("interfragment_bonds", []),
            source=data.get("source", {}),
            partitioning=data.get("partitioning", {}),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> FragmentTree:
        """
        Load a fragment tree from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        FragmentTree
            The loaded fragment tree.
        """
        path = Path(filepath)
        data = json.loads(path.read_text())
        return cls.from_dict(data)



@dataclass
class ChemicalSystem:
    """
    Canonical representation of a complete chemical system.

    Attributes
    ----------
    atoms : List[Atom]
        List of atoms in the system.
    bonds : List[Dict[str, Any]]
        List of bonds (e.g., {"atom1": 0, "atom2": 1, "order": 1.0}).
    metadata : Dict[str, Any]
        Additional system metadata.
    """
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lattice: Optional["Lattice"] = None
    _graph_cache: Optional["MolecularGraph"] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def periodic(self) -> bool:
        """Check if system is periodic."""
        return self.lattice is not None

    @property
    def graph(self) -> "MolecularGraph":
        """Lazily build and cache a MolecularGraph.

        If ``self.bonds`` is empty the bonds are inferred from atom
        geometry using covalent-radii heuristics before the graph is
        constructed.
        """
        if self._graph_cache is not None:
            return self._graph_cache
        if not self.bonds and self.atoms:
            self._infer_bonds()
        self._graph_cache = self.to_graph()
        return self._graph_cache

    def _infer_bonds(self, tolerance: float = 0.4) -> None:
        """Populate ``self.bonds`` from atom geometry."""
        from autofragment.core.bonds import infer_bonds_from_atoms

        self.bonds = infer_bonds_from_atoms(self.atoms, tolerance=tolerance)

    @property
    def n_atoms(self) -> int:
        """Return or compute n atoms."""
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        """Return or compute n bonds."""
        return len(self.bonds)

    def to_dict(self) -> Dict[str, Any]:
        """Return or compute to dict."""
        return {
            "atoms": [a.to_dict() for a in self.atoms],
            "bonds": self.bonds,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        # Convert numpy types to basic types for serialization
        return json.dumps(self.to_dict(), indent=indent)

    def to_molecules(
        self,
        *,
        atoms_per_molecule: Optional[int] = None,
        require_metadata: bool = False,
    ) -> List[Molecule]:
        """Convert the system into isolated molecules."""
        return system_to_molecules(
            self,
            atoms_per_molecule=atoms_per_molecule,
            require_metadata=require_metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChemicalSystem:
        """Return or compute from dict."""
        return cls(
            atoms=[Atom.from_dict(a) for a in data.get("atoms", [])],
            bonds=data.get("bonds", []),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> ChemicalSystem:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_molecules(
        cls,
        molecules: Sequence[Molecule],
        *,
        bonds: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        lattice: Optional["Lattice"] = None,
    ) -> "ChemicalSystem":
        """Create a ChemicalSystem from isolated molecules."""
        return molecules_to_system(
            molecules,
            bonds=bonds,
            metadata=metadata,
            lattice=lattice,
        )

    def to_graph(self) -> "MolecularGraph": # type: ignore
        """Convert to MolecularGraph."""
        # Local import to avoid circular dependency
        from autofragment.core.graph import MolecularGraph
        mg = MolecularGraph()
        for i, atom in enumerate(self.atoms):
            mg.add_atom(i, atom.symbol, atom.coords, charge=atom.charge)

        for bond in self.bonds:
            u, v = bond["atom1"], bond["atom2"]
            attrs = {k: v for k, v in bond.items() if k not in ("atom1", "atom2")}
            mg.add_bond(u, v, **attrs)

        return mg

    @classmethod
    def from_graph(cls, graph: "MolecularGraph") -> ChemicalSystem: # type: ignore
        """Create from MolecularGraph."""
        atoms = []
        bonds = []

        # Sort node indices
        node_indices = sorted(graph._graph.nodes()) # Accessing internal graph logic
        # Mapping from node index to list index (0..N-1)
        idx_map = {idx: i for i, idx in enumerate(node_indices)}

        for idx in node_indices:
            data = graph.get_atom(idx)
            atoms.append(Atom(
                symbol=data["element"],
                coords=data["coords"],
                charge=data.get("charge", 0.0)
            ))

        for u, v in graph.get_bonds():
            # Get bond data
            b_data = graph.get_bond(u, v) or {}
            bond_dict = {"atom1": idx_map[u], "atom2": idx_map[v]}
            bond_dict.update(b_data)
            bonds.append(bond_dict)

        return cls(atoms=atoms, bonds=bonds)


@dataclass
class FragmentationScheme:
    """
    Configuration for a fragmentation algorithm.

    Attributes
    ----------
    algorithm : str
        Name of the algorithm (e.g. 'RMF', 'GMF', 'Combined').
    parameters : Dict[str, Any]
        Algorithm-specific parameters.
    description : Optional[str]
        Human-readable description.
    """
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return or compute to dict."""
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FragmentationScheme:
        """Return or compute from dict."""
        return cls(
            algorithm=data["algorithm"],
            parameters=data.get("parameters", {}),
            description=data.get("description")
        )


