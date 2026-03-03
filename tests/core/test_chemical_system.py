# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ChemicalSystem."""

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem


def test_chemical_system_init():
    """Test initialization."""
    atoms = [Atom("C", np.zeros(3)), Atom("H", np.array([1., 0., 0.]))]
    bonds = [{"atom1": 0, "atom2": 1, "order": 1.0}]
    metadata = {"name": "Test"}

    sys = ChemicalSystem(atoms, bonds, metadata)
    assert sys.n_atoms == 2
    assert sys.n_bonds == 1
    assert sys.metadata["name"] == "Test"

def test_serialization():
    """Test JSON serialization."""
    atoms = [Atom("C", np.zeros(3))]
    sys = ChemicalSystem(atoms)

    json_str = sys.to_json()
    sys2 = ChemicalSystem.from_json(json_str)

    assert sys2.n_atoms == 1
    assert sys2.atoms[0].symbol == "C"
    np.testing.assert_array_equal(sys2.atoms[0].coords, atoms[0].coords)

def test_graph_conversion():
    """Test conversion to/from MolecularGraph."""
    # Create system
    atoms = [Atom("C", np.zeros(3)), Atom("H", np.array([1., 0., 0.]))]
    bonds = [{"atom1": 0, "atom2": 1, "order": 1.0}]
    sys = ChemicalSystem(atoms, bonds)

    # To graph
    mg = sys.to_graph()
    assert mg.n_atoms == 2
    assert mg.n_bonds == 1
    atom0 = mg.get_atom(0)
    assert atom0["element"] == "C"

    # From graph
    sys2 = ChemicalSystem.from_graph(mg)
    assert sys2.n_atoms == 2
    assert sys2.n_bonds == 1
    assert sys2.atoms[0].symbol == "C"

    # Check bond preservation
    # Note: from_graph remaps indices to 0..N-1, but our graph already used 0..1
    b0 = sys2.bonds[0]
    # Order might vary (u,v) or (v,u)
    assert ((b0["atom1"] == 0 and b0["atom2"] == 1) or
            (b0["atom1"] == 1 and b0["atom2"] == 0))
    assert b0["order"] == 1.0

def test_charge_preservation():
    """Test charge preservation in graph conversion."""
    atoms = [Atom("N", np.zeros(3), charge=1.0)]
    sys = ChemicalSystem(atoms)

    mg = sys.to_graph()
    atom_data = mg.get_atom(0)
    assert atom_data["charge"] == 1.0

    sys2 = ChemicalSystem.from_graph(mg)
    assert sys2.atoms[0].charge == 1.0


def test_graph_property_infers_bonds():
    """ChemicalSystem with atoms but empty bonds triggers bond inference via .graph."""
    # Water molecule: O at origin, two H atoms within bonding distance
    atoms = [
        Atom("O", np.array([0.0, 0.0, 0.0])),
        Atom("H", np.array([0.757, 0.586, 0.0])),
        Atom("H", np.array([-0.757, 0.586, 0.0])),
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=[])
    assert sys.bonds == []

    g = sys.graph
    # Bonds should have been inferred
    assert sys.n_bonds > 0
    assert g.n_atoms == 3
    assert g.n_bonds >= 2  # O-H bonds


def test_graph_property_caches():
    """Second access to .graph returns the same object."""
    atoms = [
        Atom("O", np.array([0.0, 0.0, 0.0])),
        Atom("H", np.array([0.757, 0.586, 0.0])),
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=[])
    g1 = sys.graph
    g2 = sys.graph
    assert g1 is g2


def test_graph_property_skips_inference_with_explicit_bonds():
    """Systems with explicit bonds skip inference."""
    atoms = [
        Atom("C", np.array([0.0, 0.0, 0.0])),
        Atom("H", np.array([1.09, 0.0, 0.0])),
    ]
    bonds = [{"atom1": 0, "atom2": 1, "order": 1.0}]
    sys = ChemicalSystem(atoms=atoms, bonds=bonds)

    g = sys.graph
    # Should still have exactly 1 bond (no extra inference)
    assert g.n_bonds == 1
    assert sys.n_bonds == 1

