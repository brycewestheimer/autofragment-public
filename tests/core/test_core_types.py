# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment.core.types module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autofragment.core.types import Atom, Fragment, FragmentTree


class TestAtom:
    """Tests for Atom class."""

    def test_create_atom(self):
        """Test creating an atom."""
        atom = Atom(symbol="O", coords=np.array([1.0, 2.0, 3.0]))
        assert atom.symbol == "O"
        assert np.allclose(atom.coords, [1.0, 2.0, 3.0])

    def test_create_atom_from_list(self):
        """Test creating an atom from a list of coordinates."""
        atom = Atom(symbol="H", coords=[1.0, 2.0, 3.0])
        assert isinstance(atom.coords, np.ndarray)
        assert atom.coords.shape == (3,)

    def test_atom_invalid_coords(self):
        """Test that invalid coordinates raise an error."""
        with pytest.raises(ValueError):
            Atom(symbol="O", coords=[1.0, 2.0])  # Wrong shape

    def test_atom_to_dict(self):
        """Test converting atom to dictionary."""
        atom = Atom(symbol="O", coords=np.array([1.0, 2.0, 3.0]))
        d = atom.to_dict()
        assert d["symbol"] == "O"
        assert d["coords"] == [1.0, 2.0, 3.0]

    def test_atom_from_dict(self):
        """Test creating atom from dictionary."""
        d = {"symbol": "H", "coords": [1.0, 2.0, 3.0]}
        atom = Atom.from_dict(d)
        assert atom.symbol == "H"
        assert np.allclose(atom.coords, [1.0, 2.0, 3.0])


class TestFragment:
    """Tests for Fragment class."""

    def test_create_leaf_fragment(self):
        """Test creating a leaf fragment."""
        fragment = Fragment(
            id="F1",
            symbols=["O", "H", "H"],
            geometry=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        assert fragment.id == "F1"
        assert fragment.n_atoms == 3

    def test_get_coords(self):
        """Test getting coordinates as numpy array."""
        fragment = Fragment(
            id="test",
            symbols=["O", "H"],
            geometry=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        coords = fragment.get_coords()
        assert coords.shape == (2, 3)
        assert np.allclose(coords[0], [1.0, 2.0, 3.0])
        assert np.allclose(coords[1], [4.0, 5.0, 6.0])

    def test_set_coords(self):
        """Test setting coordinates from numpy array."""
        fragment = Fragment(id="test", symbols=["O", "H"])
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fragment.set_coords(coords)
        assert fragment.geometry == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_fragment_to_dict(self):
        """Test converting fragment to dictionary."""
        fragment = Fragment(
            id="F1",
            symbols=["O", "H", "H"],
            geometry=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            molecular_charge=0,
            molecular_multiplicity=1,
        )
        d = fragment.to_dict()
        assert d["id"] == "F1"
        assert d["symbols"] == ["O", "H", "H"]

    def test_fragment_from_dict(self):
        """Test creating fragment from dictionary."""
        d = {
            "id": "F1",
            "symbols": ["O", "H", "H"],
            "geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }
        fragment = Fragment.from_dict(d)
        assert fragment.id == "F1"
        assert fragment.n_atoms == 3

    def test_fragment_from_molecules(self):
        """Test creating fragment from molecules."""
        molecules = [
            [Atom("O", [0, 0, 0]), Atom("H", [1, 0, 0]), Atom("H", [0, 1, 0])],
            [Atom("O", [5, 0, 0]), Atom("H", [6, 0, 0]), Atom("H", [5, 1, 0])],
        ]
        fragment = Fragment.from_molecules(molecules, "test")
        assert fragment.id == "test"
        assert fragment.n_atoms == 6
        assert len(fragment.symbols) == 6


class TestHierarchicalFragment:
    """Tests for hierarchical Fragment features."""

    def test_leaf_fragment_is_leaf(self):
        """Leaf fragment (no children) has is_leaf=True."""
        f = Fragment(id="F1", symbols=["O", "H"], geometry=[0] * 6)
        assert f.is_leaf
        assert f.fragments == []

    def test_nonleaf_fragment(self):
        """Non-leaf fragment has children and is_leaf=False."""
        child = Fragment(id="SF1", symbols=["O", "H"], geometry=[0] * 6)
        parent = Fragment(id="PF1", fragments=[child])
        assert not parent.is_leaf
        assert len(parent.fragments) == 1

    def test_recursive_n_atoms(self):
        """n_atoms recurses through children."""
        sf1 = Fragment(id="SF1", symbols=["O", "H", "H"], geometry=[0] * 9)
        sf2 = Fragment(id="SF2", symbols=["O", "H", "H"], geometry=[0] * 9)
        pf = Fragment(id="PF1", fragments=[sf1, sf2])
        assert pf.n_atoms == 6  # 3 + 3

    def test_hierarchical_to_dict(self):
        """to_dict includes 'fragments' key when non-empty."""
        sf = Fragment(id="SF1", symbols=["O"], geometry=[0, 0, 0])
        pf = Fragment(id="PF1", fragments=[sf])
        d = pf.to_dict()
        assert "fragments" in d
        assert len(d["fragments"]) == 1
        assert d["fragments"][0]["id"] == "SF1"

    def test_leaf_to_dict_no_fragments_key(self):
        """to_dict omits 'fragments' for leaf nodes."""
        f = Fragment(id="F1", symbols=["O"], geometry=[0, 0, 0])
        d = f.to_dict()
        assert "fragments" not in d

    def test_hierarchical_from_dict_roundtrip(self):
        """from_dict correctly deserializes nested fragments."""
        sf1 = Fragment(id="SF1", symbols=["O", "H"], geometry=[1, 2, 3, 4, 5, 6])
        sf2 = Fragment(id="SF2", symbols=["H", "H"], geometry=[7, 8, 9, 10, 11, 12])
        pf = Fragment(id="PF1", fragments=[sf1, sf2])

        d = pf.to_dict()
        restored = Fragment.from_dict(d)

        assert restored.id == "PF1"
        assert not restored.is_leaf
        assert len(restored.fragments) == 2
        assert restored.fragments[0].id == "SF1"
        assert restored.fragments[0].symbols == ["O", "H"]
        assert restored.fragments[1].id == "SF2"
        assert restored.n_atoms == 4

    def test_three_level_hierarchy(self):
        """Three levels of nesting work correctly."""
        tf = Fragment(id="TF1", symbols=["O"], geometry=[0, 0, 0])
        sf = Fragment(id="SF1", fragments=[tf])
        pf = Fragment(id="PF1", fragments=[sf])

        assert pf.n_atoms == 1
        assert not pf.is_leaf
        assert not sf.is_leaf
        assert tf.is_leaf


class TestFragmentTree:
    """Tests for FragmentTree class."""

    def test_create_tree(self):
        """Test creating a fragment tree."""
        f1 = Fragment(id="F1", symbols=["O", "H", "H"], geometry=[0] * 9)
        f2 = Fragment(id="F2", symbols=["O", "H", "H"], geometry=[0] * 9)
        tree = FragmentTree(fragments=[f1, f2])
        assert tree.n_fragments == 2

    def test_tree_to_dict(self):
        """Test converting tree to dictionary."""
        f1 = Fragment(id="F1", symbols=["O", "H", "H"], geometry=[0] * 9)
        tree = FragmentTree(
            fragments=[f1],
            source={"file": "test.xyz", "format": "xyz"},
            partitioning={"algorithm": "kmeans", "n_fragments": 1},
        )
        d = tree.to_dict()
        assert d["version"] == "2.0"
        assert d["source"]["file"] == "test.xyz"
        assert len(d["fragments"]) == 1

    def test_tree_to_json(self):
        """Test writing tree to JSON file."""
        f1 = Fragment(id="F1", symbols=["O", "H", "H"], geometry=[0] * 9)
        tree = FragmentTree(fragments=[f1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            tree.to_json(path)
            assert path.exists()

            # Read back and verify
            data = json.loads(path.read_text())
            assert data["version"] == "2.0"
            assert len(data["fragments"]) == 1

    def test_tree_from_json(self):
        """Test loading tree from JSON file."""
        f1 = Fragment(id="F1", symbols=["O", "H", "H"], geometry=[0] * 9)
        tree = FragmentTree(fragments=[f1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            tree.to_json(path)

            loaded = FragmentTree.from_json(path)
            assert loaded.n_fragments == 1
            assert loaded.fragments[0].id == "F1"

    def test_n_primary(self):
        """n_primary returns number of top-level fragments."""
        f1 = Fragment(id="F1", symbols=["O"], geometry=[0, 0, 0])
        f2 = Fragment(id="F2", symbols=["H"], geometry=[1, 1, 1])
        tree = FragmentTree(fragments=[f1, f2])
        assert tree.n_primary == 2

    def test_flat_tree_not_hierarchical(self):
        """Flat tree (all leaves) is not hierarchical."""
        f1 = Fragment(id="F1", symbols=["O"], geometry=[0, 0, 0])
        f2 = Fragment(id="F2", symbols=["H"], geometry=[1, 1, 1])
        tree = FragmentTree(fragments=[f1, f2])
        assert not tree._is_hierarchical

    def test_hierarchical_tree(self):
        """Tree with child fragments is hierarchical."""
        sf = Fragment(id="SF1", symbols=["O"], geometry=[0, 0, 0])
        pf = Fragment(id="PF1", fragments=[sf])
        tree = FragmentTree(fragments=[pf])
        assert tree._is_hierarchical

    def test_hierarchical_n_fragments(self):
        """n_fragments counts all levels for hierarchical tree."""
        sf1 = Fragment(id="SF1", symbols=["O"], geometry=[0, 0, 0])
        sf2 = Fragment(id="SF2", symbols=["H"], geometry=[1, 1, 1])
        pf = Fragment(id="PF1", fragments=[sf1, sf2])
        tree = FragmentTree(fragments=[pf])
        # 1 primary + 2 secondary = 3
        assert tree.n_fragments == 3

    def test_hierarchical_n_atoms(self):
        """n_atoms sums recursively for hierarchical tree."""
        sf1 = Fragment(id="SF1", symbols=["O", "H"], geometry=[0] * 6)
        sf2 = Fragment(id="SF2", symbols=["H", "H"], geometry=[0] * 6)
        pf = Fragment(id="PF1", fragments=[sf1, sf2])
        tree = FragmentTree(fragments=[pf])
        assert tree.n_atoms == 4

    def test_hierarchical_json_roundtrip(self):
        """Hierarchical tree survives JSON roundtrip."""
        sf1 = Fragment(id="SF1", symbols=["O"], geometry=[1, 2, 3])
        sf2 = Fragment(id="SF2", symbols=["H"], geometry=[4, 5, 6])
        pf = Fragment(id="PF1", fragments=[sf1, sf2])
        tree = FragmentTree(fragments=[pf])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            tree.to_json(path)

            loaded = FragmentTree.from_json(path)
            assert loaded._is_hierarchical
            assert loaded.n_fragments == 3
            assert loaded.n_atoms == 2
            assert loaded.fragments[0].id == "PF1"
            assert loaded.fragments[0].fragments[0].id == "SF1"
