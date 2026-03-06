# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-based molecular representation using networkx."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from autofragment.core.types import FragmentTree, Molecule
from autofragment.optional import require_dependency

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

from autofragment.core.bonds import infer_bonds_from_atoms, is_bonded
from autofragment.core.chemistry import infer_bond_order


class MolecularGraph:

    """Graph representation of a molecular system.

    Atoms are represented as nodes with attributes:
    - element: str (e.g., 'C', 'H', 'O')
    - coords: np.ndarray of shape (3,)
    - charge: float (optional)

    Bonds are represented as edges with attributes:
    - order: float (1.0, 1.5, 2.0, 3.0)
    - bond_type: str (e.g., 'single', 'double', 'aromatic')

    Example:
        >>> mg = MolecularGraph()
        >>> mg.add_atom(0, "O", np.array([0., 0., 0.]))
        >>> mg.add_atom(1, "H", np.array([0.7, 0.6, 0.]))
        >>> mg.add_atom(2, "H", np.array([-0.7, 0.6, 0.]))
        >>> mg.add_bond(0, 1, order=1.0)
        >>> mg.add_bond(0, 2, order=1.0)
        >>> mg.n_atoms
        3
    """

    def __init__(self, graph: Optional["nx.Graph"] = None):
        """Initialize the molecular graph.

        Args:
            graph: Optional existing networkx Graph to wrap. If None, creates a new one.
        """
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        self._graph: Any = graph if graph is not None else nx.Graph()

    @classmethod
    def from_molecules(
        cls,
        molecules: List[Molecule],
        infer_bonds: bool = True,
        tolerance: float = 0.4,
    ) -> "MolecularGraph":
        """Create graph from list of molecules.

        Parameters
        ----------
        molecules : List[Molecule]
            List of molecules (each is List[Atom])
        infer_bonds : bool
            If True, detect bonds from geometry
        tolerance : float
            Bond detection tolerance in Angstroms
        """
        mg = cls()
        current_idx = 0

        # Add all atoms
        for mol in molecules:
            for atom in mol:
                # Access attributes from Atom dataclass
                mg.add_atom(
                    current_idx,
                    element=atom.symbol,
                    coords=atom.coords,
                    charge=atom.charge
                )
                current_idx += 1

        # Infer bonds if requested
        if infer_bonds and mg.n_atoms > 1:
            all_atoms = [atom for mol in molecules for atom in mol]
            raw_bonds = infer_bonds_from_atoms(all_atoms, tolerance=tolerance)
            for bond in raw_bonds:
                i, j = bond["atom1"], bond["atom2"]
                atom1 = mg.get_atom(i)
                atom2 = mg.get_atom(j)
                dist = float(np.linalg.norm(atom1["coords"] - atom2["coords"]))
                order = infer_bond_order(atom1["element"], atom2["element"], dist)
                if order == 0.0:
                    order = 1.0
                mg.add_bond(i, j, order=order)

        return mg

    @classmethod
    def from_fragment_tree(
        cls,
        tree: FragmentTree,
        infer_missing_bonds: bool = True,
        tolerance: float = 0.4,
    ) -> "MolecularGraph":
        """Reconstruct molecular graph from a fragment tree.

        Preserves fragment membership as an atom attribute 'fragment_id'.
        Reconstructs interfragment bonds from the tree's store.
        If infer_missing_bonds is True, infers bonds within fragments.
        """
        mg = cls()
        current_idx = 0

        # We need to map fragment-local atom indices to global graph indices
        # Key: (fragment_id, local_index), Value: global_index
        atom_map: Dict[Tuple[str, int], int] = {}

        for frag in tree.fragments:
            coords = frag.get_coords()
            symbols = frag.symbols
            n = len(symbols)

            # Map atoms
            for local_i, symbol in enumerate(symbols):
                global_i = current_idx
                atom_map[(frag.id, local_i)] = global_i

                mg.add_atom(
                    global_i,
                    element=symbol,
                    coords=coords[local_i],
                    fragment_id=frag.id,
                    charge=0.0 # TODO: Charge not stored in Fragment atoms explicitly
                )
                current_idx += 1

            # Infer bonds within this fragment
            if infer_missing_bonds and n > 1:
                for i in range(n):
                    u = atom_map[(frag.id, i)]
                    for j in range(i + 1, n):
                        v = atom_map[(frag.id, j)]

                        # Check connectivity
                        if is_bonded(
                            symbols[i], tuple(coords[i]),
                            symbols[j], tuple(coords[j]),
                            tolerance=tolerance
                        ):
                            dist = float(np.linalg.norm(coords[i] - coords[j]))
                            order = infer_bond_order(symbols[i], symbols[j], dist)
                            if order == 0.0:
                                order = 1.0
                            mg.add_bond(u, v, order=order)

        # Add interfragment bonds stored in the tree
        for bond in tree.interfragment_bonds:
            fid1 = bond["fragment1_id"]
            idx1 = bond["atom1_index"]
            fid2 = bond["fragment2_id"]
            idx2 = bond["atom2_index"]
            order = bond.get("bond_order", 1.0)
            metadata = bond.get("metadata", {})

            if (fid1, idx1) in atom_map and (fid2, idx2) in atom_map:
                u = atom_map[(fid1, idx1)]
                v = atom_map[(fid2, idx2)]

                # Interfragment bonds are explicitly defined
                mg.add_bond(u, v, order=order, is_interfragment=True, **metadata)

        return mg



    @property
    def networkx_graph(self) -> "nx.Graph":
        """Read-only access to the underlying NetworkX graph.

        Use this when algorithm code needs direct access to the networkx
        graph object (e.g. for graph partitioning routines).  Prefer the
        public methods on ``MolecularGraph`` when possible.
        """
        return self._graph

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the graph."""
        return self._graph.number_of_nodes()

    @property
    def n_bonds(self) -> int:
        """Number of bonds in the graph."""
        return self._graph.number_of_edges()

    def nodes(self) -> List[int]:
        """Return a sorted list of all node (atom) indices."""
        return sorted(self._graph.nodes())

    def neighbors(self, index: int) -> List[int]:
        """Return a list of neighbor indices for the given atom.

        Args:
            index: Atom index.

        Returns:
            List of neighbor atom indices.
        """
        return list(self._graph.neighbors(index))

    def has_edge(self, atom1: int, atom2: int) -> bool:
        """Check if a bond exists between two atoms.

        Args:
            atom1: First atom index.
            atom2: Second atom index.

        Returns:
            True if a bond exists.
        """
        return self._graph.has_edge(atom1, atom2)

    def remove_edge(self, atom1: int, atom2: int) -> None:
        """Remove the bond between two atoms.

        Args:
            atom1: First atom index.
            atom2: Second atom index.

        Raises:
            networkx.NetworkXError: If the bond does not exist.
        """
        self._graph.remove_edge(atom1, atom2)

    def is_connected(self) -> bool:
        """Check if the graph is connected.

        Returns:
            True if the graph is connected.
        """
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        return nx.is_connected(self._graph)

    def add_atom(self, index: int, element: str, coords: np.ndarray, **attrs: Any) -> None:
        """Add an atom to the graph.

        Args:
            index: Unique integer index for the atom.
            element: Atomic symbol (e.g., 'C', 'H').
            coords: Cartesian coordinates as structure (x, y, z).
            **attrs: Additional attributes (e.g., charge).
        """
        self._graph.add_node(index, element=element, coords=coords, **attrs)

    def add_bond(self, atom1: int, atom2: int, order: float = 1.0, **attrs: Any) -> None:
        """Add a bond between two atoms.

        Args:
            atom1: Index of the first atom.
            atom2: Index of the second atom.
            order: Bond order (default 1.0). Valid values: {1.0, 1.5, 2.0, 3.0}.
            **attrs: Additional attributes (e.g., bond_type, rotatable).

        Raises:
            ValueError: If bond order is invalid.
        """
        valid_orders = {1.0, 1.5, 2.0, 3.0}
        if order not in valid_orders:
            raise ValueError(f"Invalid bond order: {order}. Must be one of {valid_orders}")

        self._graph.add_edge(atom1, atom2, order=order, **attrs)

    def get_bonds(self) -> List[Tuple[int, int]]:
        """Get all bonds as a list of tuples (atom1, atom2).

        Returns:
             List of (u, v) tuples representing edges.
        """
        return list(self._graph.edges())


    def get_atom(self, index: int) -> Dict[str, Any]:
        """Get atom attributes.

        Args:
            index: Atom index.

        Returns:
            Dictionary of atom attributes.

        Raises:
            KeyError: If atom index does not exist.
        """
        return self._graph.nodes[index] # type: ignore

    def get_bond(self, atom1: int, atom2: int) -> Optional[Dict[str, Any]]:
        """Get bond attributes, or None if no bond exists.

        Args:
            atom1: Index of the first atom.
            atom2: Index of the second atom.

        Returns:
            Dictionary of bond attributes if bond exists, else None.
        """
        if self._graph.has_edge(atom1, atom2):
            return self._graph.edges[atom1, atom2] # type: ignore
        return None

    def get_coordinates(self) -> np.ndarray:
        """Get coordinates of all atoms as an Nx3 array.

        Returns:
            Nx3 numpy array of coordinates, ordered by node index.
        """
        if self.n_atoms == 0:
            return np.zeros((0, 3))

        # Get all nodes with 'coords' attribute, sorted by index
        coords_list = [self._graph.nodes[i]["coords"] for i in sorted(self._graph.nodes())]
        return np.array(coords_list)

    def connected_components(self) -> List[Set[int]]:
        """Find connected components (separate molecules).

        Returns:
            List of sets, where each set contains node indices of a component.
        """
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        return [set(c) for c in nx.connected_components(self._graph)]

    @property
    def n_components(self) -> int:
        """Number of connected components."""
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        return nx.number_connected_components(self._graph)

    def find_rings(self) -> List[List[int]]:
        """Find all relevant rings (fundamental cycle basis).

        Returns:
            List of rings, where each ring is a list of atom indices.
        """
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        return nx.cycle_basis(self._graph)

    def is_in_ring(self, atom1: int, atom2: int) -> bool:
        """Check if the bond between atom1 and atom2 is in a ring.

        Args:
            atom1: First atom index.
            atom2: Second atom index.

        Returns:
            True if the bond is part of a ring.
        """
        if not self._graph.has_edge(atom1, atom2):
            return False

        # Check if there is an alternative path
        # We temporarily remove the edge and check for connectivity
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        attrs = self._graph.edges[atom1, atom2]
        self._graph.remove_edge(atom1, atom2)
        try:
            return nx.has_path(self._graph, atom1, atom2)
        finally:
            self._graph.add_edge(atom1, atom2, **attrs)

    def find_bridges(self) -> List[Tuple[int, int]]:
        """Find all bridge bonds (cut-edges).

        Returns:
            List of (u, v) tuples.
        """
        nx = require_dependency("networkx", "graph", "MolecularGraph")
        return list(nx.bridges(self._graph))

    def is_bridge(self, atom1: int, atom2: int) -> bool:
        """Check if a bond is a bridge.

        Args:
            atom1: First atom index.
            atom2: Second atom index.

        Returns:
            True if the bond is a bridge.
        """
        if not self._graph.has_edge(atom1, atom2):
            return False

        # A bridge is an edge whose removal increases the number of connected components.
        # Equivalent to: no alternative path exists.
        # This is the exact inverse of being in a ring (for an existing edge).
        return not self.is_in_ring(atom1, atom2)

    def subgraph(self, atom_indices: Set[int]) -> "MolecularGraph":
        """Extract subgraph containing specified atoms.

        Args:
            atom_indices: Set of atom indices to include.

        Returns:
            New MolecularGraph containing only the specified atoms and induced edges.
            The returned graph is an independent copy.
        """
        # Create an independent copy of the induced subgraph
        sub = self._graph.subgraph(atom_indices).copy()

        # We need to wrap it in a new MolecularGraph
        # Note: MolecularGraph constructor takes nx.Graph
        # If we pass arguments to constructor it works
        mg = MolecularGraph(graph=sub)
        return mg

    def subgraph_by_attribute(self, attr: str, value: Any) -> "MolecularGraph":
        """Extract subgraph where nodes have matching attribute.

        Args:
            attr: Attribute name (e.g. 'fragment_id').
            value: Attribute value (e.g. 'F1').

        Returns:
            New MolecularGraph containing matching atoms.
        """
        indices = {n for n, d in self._graph.nodes(data=True) if d.get(attr) == value}
        return self.subgraph(indices)


class SparseBondGraph:
    """Sparse adjacency matrix for bonds using scipy (optional)."""

    def __init__(self, n_atoms: int):
        """Initialize a new SparseBondGraph instance."""
        scipy = require_dependency("scipy", "graph", "SparseBondGraph")
        self._adjacency = scipy.sparse.lil_matrix((n_atoms, n_atoms), dtype=np.int8)

    def add_bond(self, i: int, j: int, order: int = 1) -> None:
        """Add a bond to the sparse adjacency matrix."""
        self._adjacency[i, j] = order
        self._adjacency[j, i] = order

    @property
    def adjacency(self):
        """Return the underlying sparse adjacency matrix."""
        return self._adjacency





