# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MolecularGraph class."""

import networkx as nx
import numpy as np
import pytest

from autofragment.core.graph import MolecularGraph
from autofragment.core.types import Atom, Fragment, FragmentTree


def test_from_molecules():
    """Test from_molecules factory method."""
    # Water: O at origin, H1 at (0, 0.76, 0.6), H2 at (0, -0.76, 0.6)
    # Using simple coordinates for bond detection
    # O (0,0,0)
    # H (0.9, 0, 0) -> dist 0.9. O (0.66) + H (0.31) = 0.97. Bonded.
    # H (-0.9, 0, 0)

    o_coords = np.array([0.0, 0.0, 0.0])
    h1_coords = np.array([0.9, 0.0, 0.0])
    h2_coords = np.array([-0.9, 0.0, 0.0])

    mol1 = [
        Atom("O", o_coords, charge=-0.8),
        Atom("H", h1_coords, charge=0.4),
        Atom("H", h2_coords, charge=0.4)
    ]

    # Another separate molecule (far away)
    c_coords = np.array([10.0, 0.0, 0.0])
    mol2 = [
        Atom("C", c_coords)
    ]

    molecules = [mol1, mol2]

    # Test with inference
    mg = MolecularGraph.from_molecules(molecules, infer_bonds=True)

    assert mg.n_atoms == 4
    assert mg.n_bonds == 2 # O-H1, O-H2

    # Check atoms
    atom0 = mg.get_atom(0)
    assert atom0["element"] == "O"
    assert atom0["charge"] == -0.8

    # Check bonds (0-1 and 0-2)
    assert mg.get_bond(0, 1) is not None
    assert mg.get_bond(0, 2) is not None
    assert mg.get_bond(1, 2) is None # H-H not bonded

    # Test without inference
    mg_no_bonds = MolecularGraph.from_molecules(molecules, infer_bonds=False)
    assert mg_no_bonds.n_atoms == 4
    assert mg_no_bonds.n_bonds == 0


def test_initialization():
    """Test initialization with and without existing graph."""
    # Empty init
    mg = MolecularGraph()
    assert mg.n_atoms == 0
    assert mg.n_bonds == 0

    # Init with existing graph
    g = nx.Graph()
    g.add_node(0, element="C")
    mg2 = MolecularGraph(g)
    assert mg2.n_atoms == 1

def test_add_atom():
    """Test adding atoms."""
    mg = MolecularGraph()
    coords = np.array([0.0, 0.0, 0.0])
    mg.add_atom(0, "C", coords, charge=0.5)

    assert mg.n_atoms == 1
    atom_data = mg.get_atom(0)
    assert atom_data["element"] == "C"
    np.testing.assert_array_equal(atom_data["coords"], coords)
    assert atom_data["charge"] == 0.5

def test_add_bond():
    """Test adding bonds."""
    mg = MolecularGraph()
    coords = np.array([0.0, 0.0, 0.0])
    mg.add_atom(0, "C", coords)
    mg.add_atom(1, "H", coords)

    mg.add_bond(0, 1, order=1.5, bond_type="aromatic")

    assert mg.n_bonds == 1
    bond_data = mg.get_bond(0, 1)
    assert bond_data is not None
    assert bond_data["order"] == 1.5
    assert bond_data["bond_type"] == "aromatic"

    # Test non-existent bond
    assert mg.get_bond(0, 2) is None

def test_get_atom_error():
    """Test getting non-existent atom raises error."""
    mg = MolecularGraph()
    with pytest.raises(KeyError):
        mg.get_atom(99)

def test_properties():
    """Test n_atoms and n_bonds properties."""
    mg = MolecularGraph()
    assert mg.n_atoms == 0
    assert mg.n_bonds == 0

    mg.add_atom(0, "C", np.zeros(3))
    assert mg.n_atoms == 1
    assert mg.n_bonds == 0

    mg.add_atom(1, "H", np.zeros(3))

    mg.add_bond(0, 1)
    assert mg.n_atoms == 2
    assert mg.n_bonds == 1

def test_node_attributes():
    """Test setting and getting various node attributes."""
    mg = MolecularGraph()
    coords = np.array([1., 2., 3.])
    mg.add_atom(0, "C", coords, charge=-0.5, atom_type="CA", residue="ALA", chain="A")

    atom = mg.get_atom(0)
    assert atom["element"] == "C"
    np.testing.assert_array_equal(atom["coords"], coords)
    assert atom["charge"] == -0.5
    assert atom["atom_type"] == "CA"
    assert atom["residue"] == "ALA"
    assert atom["chain"] == "A"

def test_get_coordinates():
    """Test retrieving all coordinates."""
    mg = MolecularGraph()

    # Test empty
    assert mg.get_coordinates().shape == (0, 3)

    # Add atoms
    c1 = np.array([0., 0., 0.])
    c2 = np.array([1., 1., 1.])
    c3 = np.array([2., 2., 2.])

    # Add out of order to verify sorting
    mg.add_atom(0, "H", c1)
    mg.add_atom(2, "H", c3)
    mg.add_atom(1, "O", c2)

    coords = mg.get_coordinates()
    assert coords.shape == (3, 3)
    np.testing.assert_array_equal(coords[0], c1)
    np.testing.assert_array_equal(coords[1], c2)
    np.testing.assert_array_equal(coords[2], c3)

def test_edge_attributes():
    """Test setting and getting various edge attributes."""
    mg = MolecularGraph()
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "C", np.zeros(3))

    mg.add_bond(0, 1, order=2.0, bond_type="double", rotatable=False, ring_member=True)

    bond = mg.get_bond(0, 1)
    assert bond["order"] == 2.0
    assert bond["bond_type"] == "double"
    assert bond["rotatable"] is False
    assert bond["ring_member"] is True

def test_bond_order_validation():
    """Test that invalid bond orders are rejected."""
    mg = MolecularGraph()
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "C", np.zeros(3))

    with pytest.raises(ValueError, match="Invalid bond order"):
        mg.add_bond(0, 1, order=5.0)

def test_get_bonds():
    """Test retrieving all bonds."""
    mg = MolecularGraph()
    mg.add_atom(0, "A", np.zeros(3))
    mg.add_atom(1, "B", np.zeros(3))
    mg.add_atom(2, "C", np.zeros(3))

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)

    bonds = mg.get_bonds()
    assert len(bonds) == 2

    bond_set = {tuple(sorted(b)) for b in bonds}
    assert (0, 1) in bond_set
    assert (1, 2) in bond_set

def test_connected_components():
    """Test connected components detection."""
    mg = MolecularGraph()
    assert mg.n_components == 0
    assert mg.connected_components() == []

    # Molecule 1: 0-1
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "O", np.zeros(3))
    mg.add_bond(0, 1)

    assert mg.n_components == 1
    comps = mg.connected_components()
    assert len(comps) == 1
    assert comps[0] == {0, 1}

    # Molecule 2: 2-3-4
    mg.add_atom(2, "H", np.zeros(3))
    mg.add_atom(3, "O", np.zeros(3))
    mg.add_atom(4, "H", np.zeros(3))
    mg.add_bond(2, 3)
    mg.add_bond(3, 4)

    assert mg.n_components == 2
    comps = mg.connected_components()
    assert len(comps) == 2

    # We don't know the order, so check if sets are present
    comp_sets = [c for c in comps]
    assert {0, 1} in comp_sets
    assert {2, 3, 4} in comp_sets

def test_find_rings():
    """Test ring detection."""
    mg = MolecularGraph()
    # Simple cycle: 0-1-2-0
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "C", np.zeros(3))
    mg.add_atom(2, "C", np.zeros(3))

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)
    mg.add_bond(2, 0)

    rings = mg.find_rings()
    assert len(rings) == 1
    assert set(rings[0]) == {0, 1, 2}

    # Fused cycle: square 0-1-2-3 with diagonal 1-3
    # 0-1-2-3-0 is a square.
    # 1-3 cuts it into two triangles: 0-1-3 and 1-2-3.
    # 0 -- 1
    # |  / |
    # 3 -- 2
    mg = MolecularGraph()
    for i in range(4):
        mg.add_atom(i, "C", np.zeros(3))

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)
    mg.add_bond(2, 3)
    mg.add_bond(3, 0)
    mg.add_bond(1, 3)

    rings = mg.find_rings()
    assert len(rings) == 2
    # Usually basis is the two triangles, or one triangle + the square.
    # But networkx cycle basis usually selects smallest cycles for planar graphs.

def test_is_in_ring():
    """Test checking if bond is in ring."""
    mg = MolecularGraph()
    # 0-1-2-0 triangle with tail 2-3
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "C", np.zeros(3))
    mg.add_atom(2, "C", np.zeros(3))
    mg.add_atom(3, "C", np.zeros(3))

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)
    mg.add_bond(2, 0)
    mg.add_bond(2, 3)

    # Triangle edges
    assert mg.is_in_ring(0, 1)
    assert mg.is_in_ring(1, 2)
    assert mg.is_in_ring(0, 2)

    # Tail edge
    assert not mg.is_in_ring(2, 3)

    # Non-existent edge
    assert not mg.is_in_ring(0, 3)

def test_find_bridges():
    """Test finding bridges (cut-edges)."""
    mg = MolecularGraph()
    # 0-1 (bridge) - 2-3-4-2 (triangle on 2)
    for i in range(5):
        mg.add_atom(i, "C", np.zeros(3))

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)
    mg.add_bond(2, 3)
    mg.add_bond(3, 4)
    mg.add_bond(4, 2)

    bridges = mg.find_bridges()

    # 0-1 and 1-2 are bridges.
    # 2-3, 3-4, 4-2 are in cycle.

    assert len(bridges) == 2
    b_set = {tuple(sorted(b)) for b in bridges}
    assert (0, 1) in b_set
    assert (1, 2) in b_set
    assert (2, 3) not in b_set

def test_is_bridge():
    """Test checking if specific bond is a bridge."""
    mg = MolecularGraph()
    # 0-1
    mg.add_atom(0, "C", np.zeros(3))
    mg.add_atom(1, "C", np.zeros(3))
    mg.add_bond(0, 1)

    assert mg.is_bridge(0, 1)

    # Add 1-2
    mg.add_atom(2, "C", np.zeros(3))
    mg.add_bond(1, 2)
    assert mg.is_bridge(0, 1)
    assert mg.is_bridge(1, 2)

    # Close loop 2-0 -> 0-1-2-0
    mg.add_bond(2, 0)
    assert not mg.is_bridge(0, 1) # Ring
    assert not mg.is_bridge(1, 2) # Ring
    assert not mg.is_bridge(2, 0) # Ring

    assert not mg.is_bridge(0, 99)

def test_from_fragment_tree():
    """Test generating graph from FragmentTree."""
    # F1: C-C (0,1)
    # F2: O (2)

    # Coordinates: 0 at origin, 1 at (1.5, 0, 0), 2 at (3.0, 0, 0)
    # 0-1 dist 1.5 (C-C ~1.54) -> should be bonded
    # 1-2 dist 1.5 (C-O ~1.43) -> should be bonded BUT we define it as interfrag

    f1 = Fragment(
        id="F1",
        symbols=["C", "C"],
        geometry=[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    )
    f2 = Fragment(
        id="F2",
        symbols=["O"],
        geometry=[3.0, 0.0, 0.0]
    )

    bond = {
        "fragment1_id": "F1",
        "atom1_index": 1,
        "fragment2_id": "F2",
        "atom2_index": 0,
        "bond_order": 1.0,
        "metadata": {"type": "covalent"}
    }

    tree = FragmentTree(fragments=[f1, f2], interfragment_bonds=[bond])

    mg = MolecularGraph.from_fragment_tree(tree, infer_missing_bonds=True)

    assert mg.n_atoms == 3
    # Check inferred bond within F1
    # Global indices: 0(F1,0), 1(F1,1), 2(F2,0)
    assert mg.get_bond(0, 1) is not None
    # Check explicit interfrag bond
    assert mg.get_bond(1, 2) is not None
    assert mg.get_bond(1, 2)["is_interfragment"] is True
    assert mg.get_bond(1, 2)["type"] == "covalent"

    # Check node attrs
    assert mg.get_atom(0)["fragment_id"] == "F1"
    assert mg.get_atom(2)["fragment_id"] == "F2"

def test_subgraph():
    """Test subgraph extraction."""
    # 0-1-2-3 (linear)
    mg = MolecularGraph()
    for i in range(4):
        mg.add_atom(i, "C", np.zeros(3), fragment_id=1 if i < 2 else 2)

    mg.add_bond(0, 1)
    mg.add_bond(1, 2)
    mg.add_bond(2, 3)

    # Subgraph {0, 1}
    sub = mg.subgraph({0, 1})
    assert sub.n_atoms == 2
    assert sub.n_bonds == 1 # 0-1 is preserved
    assert sub.get_atom(0)["fragment_id"] == 1

    # Check induced (1-2 is not in subgraph since 2 is missing)
    assert sub.get_bond(1, 2) is None

    # Subgraph by attribute
    sub2 = mg.subgraph_by_attribute("fragment_id", 2)
    assert sub2.n_atoms == 2 # 2, 3

    # Nodes are 2 and 3. Indices preserved.
    assert sub2.get_atom(2)["fragment_id"] == 2
    assert sub2.get_atom(3)["fragment_id"] == 2
    assert sub2.n_bonds == 1 # 2-3

    # Check independence
    sub2.add_atom(99, "X", np.zeros(3))
    assert mg.n_atoms == 4 # Original untouched







