# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for QM/MM partitioner."""

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem
from autofragment.multilevel import LinkAtom
from autofragment.partitioners.qmmm import (
    AtomSelection,
    CombinedSelection,
    DistanceSelection,
    QMMMPartitioner,
    QMMMResult,
    ResidueSelection,
    TopologySelection,
)


class TestQMMMResult:
    """Tests for QMMMResult dataclass."""

    def test_result_creation(self):
        """Test creating a QM/MM result."""
        result = QMMMResult(
            qm_atoms={0, 1, 2},
            buffer_atoms={3, 4},
            mm_atoms={5, 6, 7, 8, 9},
        )
        assert result.n_qm_atoms == 3
        assert result.n_buffer_atoms == 2
        assert result.n_mm_atoms == 5
        assert result.total_atoms == 10

    def test_result_with_link_atoms(self):
        """Test result with link atoms."""
        link = LinkAtom(qm_atom_index=2, mm_atom_index=3)
        result = QMMMResult(
            qm_atoms={0, 1, 2},
            buffer_atoms=set(),
            mm_atoms={3, 4, 5},
            link_atoms=[link],
            cut_bonds=[(2, 3)],
        )
        assert result.n_link_atoms == 1
        assert result.cut_bonds == [(2, 3)]

    def test_to_multilevel_scheme(self):
        """Test conversion to MultiLevelScheme."""
        result = QMMMResult(
            qm_atoms={0, 1, 2},
            buffer_atoms={3, 4},
            mm_atoms={5, 6, 7},
            qm_charge=-1,
            qm_multiplicity=2,
        )
        scheme = result.to_multilevel_scheme(
            qm_method="B3LYP",
            qm_basis="6-31G*",
            mm_method="AMBER",
        )
        assert scheme.scheme_type == "qmmm"
        assert scheme.n_layers == 3  # QM, buffer, MM

        qm_layer = scheme.get_layer("qm")
        assert qm_layer.method == "B3LYP"
        assert qm_layer.charge == -1
        assert qm_layer.multiplicity == 2

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        link = LinkAtom(qm_atom_index=2, mm_atom_index=3)
        result = QMMMResult(
            qm_atoms={0, 1, 2},
            buffer_atoms={3, 4},
            mm_atoms={5, 6},
            link_atoms=[link],
            cut_bonds=[(2, 3)],
            qm_charge=1,
            qm_multiplicity=2,
        )

        d = result.to_dict()
        rebuilt = QMMMResult.from_dict(d)

        assert rebuilt.qm_atoms == result.qm_atoms
        assert rebuilt.buffer_atoms == result.buffer_atoms
        assert rebuilt.mm_atoms == result.mm_atoms
        assert len(rebuilt.link_atoms) == 1
        assert rebuilt.cut_bonds == result.cut_bonds


class TestAtomSelection:
    """Tests for atom-based selection."""

    def test_basic_selection(self):
        """Test simple atom selection."""
        selection = AtomSelection(atom_indices={0, 1, 2})
        coords = np.zeros((10, 3))
        elements = ["C"] * 10

        selected = selection.select(coords, elements)
        assert selected == {0, 1, 2}

    def test_selection_returns_copy(self):
        """Test that selection returns a copy."""
        selection = AtomSelection(atom_indices={0, 1})
        coords = np.zeros((5, 3))
        elements = ["C"] * 5

        selected = selection.select(coords, elements)
        selected.add(10)  # Modify returned set

        # Original should be unchanged
        assert selection.atom_indices == {0, 1}


class TestDistanceSelection:
    """Tests for distance-based selection."""

    def test_basic_distance_selection(self):
        """Test distance-based selection."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        elements = ["C"] * 4

        selection = DistanceSelection(center=np.array([0, 0, 0]), radius=2.5)
        selected = selection.select(coords, elements)

        assert selected == {0, 1, 2}
        assert 3 not in selected  # Too far

    def test_empty_selection(self):
        """Test when no atoms are within radius."""
        coords = np.array([[10.0, 0, 0], [20.0, 0, 0]])
        elements = ["C", "C"]

        selection = DistanceSelection(center=np.array([0, 0, 0]), radius=1.0)
        selected = selection.select(coords, elements)

        assert selected == set()


class TestResidueSelection:
    """Tests for residue-based selection."""

    def test_selection_by_name(self):
        """Test selection by residue name."""
        selection = ResidueSelection(residue_names=["HIS", "CYS"])
        coords = np.zeros((5, 3))
        elements = ["N", "C", "C", "S", "O"]

        selected = selection.select(
            coords, elements,
            residue_names=["HIS", "HIS", "GLY", "CYS", "WAT"]
        )

        assert selected == {0, 1, 3}

    def test_selection_by_number(self):
        """Test selection by residue number."""
        selection = ResidueSelection(residue_numbers=[1, 3])
        coords = np.zeros((6, 3))
        elements = ["C"] * 6

        selected = selection.select(
            coords, elements,
            residue_numbers=[1, 1, 2, 2, 3, 3]
        )

        assert selected == {0, 1, 4, 5}


class TestCombinedSelection:
    """Tests for combined selection strategies."""

    def test_union_selection(self):
        """Test union of selections."""
        sel1 = AtomSelection({0, 1})
        sel2 = AtomSelection({2, 3})

        combined = CombinedSelection([sel1, sel2], mode="union")
        coords = np.zeros((10, 3))
        elements = ["C"] * 10

        selected = combined.select(coords, elements)
        assert selected == {0, 1, 2, 3}

    def test_intersection_selection(self):
        """Test intersection of selections."""
        sel1 = AtomSelection({0, 1, 2, 3})
        sel2 = AtomSelection({2, 3, 4, 5})

        combined = CombinedSelection([sel1, sel2], mode="intersection")
        coords = np.zeros((10, 3))
        elements = ["C"] * 10

        selected = combined.select(coords, elements)
        assert selected == {2, 3}


class TestQMMMPartitioner:
    """Tests for QMMMPartitioner."""

    def create_linear_system(self, n_atoms: int) -> ChemicalSystem:
        """Create a linear chain of atoms for testing."""
        atoms = []
        for i in range(n_atoms):
            atom = Atom(symbol="C", coords=np.array([float(i) * 1.5, 0.0, 0.0]))
            atoms.append(atom)
        return ChemicalSystem(atoms=atoms)

    def test_basic_partition(self):
        """Test basic QM/MM partitioning."""
        system = self.create_linear_system(10)
        selection = AtomSelection({0, 1, 2})

        partitioner = QMMMPartitioner(
            qm_selection=selection,
            buffer_radius=3.0,  # Will get atom 3
        )

        # Provide bonds for the linear chain
        bonds = [(i, i+1) for i in range(9)]
        result = partitioner.partition(system, bonds=bonds)

        assert result.qm_atoms == {0, 1, 2}
        assert 3 in result.buffer_atoms  # Within 3A of QM
        # Cut bond should be at QM/MM boundary
        assert (2, 3) in result.cut_bonds

    def test_no_buffer(self):
        """Test partition with no buffer."""
        system = self.create_linear_system(5)
        selection = AtomSelection({0, 1})

        partitioner = QMMMPartitioner(
            qm_selection=selection,
            buffer_radius=0.0,
        )

        bonds = [(i, i+1) for i in range(4)]
        result = partitioner.partition(system, bonds=bonds)

        assert result.buffer_atoms == set()
        assert result.mm_atoms == {2, 3, 4}

    def test_link_atoms_positioned(self):
        """Test that link atoms are correctly positioned."""
        system = self.create_linear_system(5)
        selection = AtomSelection({0, 1})

        partitioner = QMMMPartitioner(
            qm_selection=selection,
            buffer_radius=0.0,
            link_scheme="hydrogen",
        )

        bonds = [(i, i+1) for i in range(4)]
        result = partitioner.partition(system, bonds=bonds)

        assert len(result.link_atoms) == 1
        link = result.link_atoms[0]
        assert link.qm_atom_index == 1
        assert link.mm_atom_index == 2
        assert link.element == "H"
        assert link.position is not None

    def test_charge_and_multiplicity(self):
        """Test passing charge and multiplicity."""
        system = self.create_linear_system(3)
        selection = AtomSelection({0, 1})

        partitioner = QMMMPartitioner(qm_selection=selection, buffer_radius=0)
        bonds = [(0, 1), (1, 2)]
        result = partitioner.partition(
            system, bonds=bonds,
            qm_charge=-1,
            qm_multiplicity=2,
        )

        assert result.qm_charge == -1
        assert result.qm_multiplicity == 2

    def test_partition_to_fragment_tree(self):
        """Test conversion to FragmentTree."""
        system = self.create_linear_system(5)
        selection = AtomSelection({0, 1})

        partitioner = QMMMPartitioner(qm_selection=selection, buffer_radius=0)
        bonds = [(i, i+1) for i in range(4)]

        tree = partitioner.partition_to_fragment_tree(system, bonds=bonds)

        assert tree.n_fragments == 2
        assert tree.fragments[0].id == "QM"
        assert tree.fragments[1].id == "MM"

    def test_topology_selection_graph_hops(self):
        """Test topology-based QM selection using graph hops."""
        system = self.create_linear_system(6)
        selection = TopologySelection(seed_atoms={0}, mode="graph", hops=2)

        partitioner = QMMMPartitioner(
            qm_selection=selection,
            buffer_radius=0.0,
        )

        bonds = [(i, i + 1) for i in range(5)]
        result = partitioner.partition(system, bonds=bonds)

        assert result.qm_atoms == {0, 1, 2}


class TestBondDetection:
    """Tests for bond detection in partitioner."""

    def test_detect_bonds(self):
        """Test automatic bond detection."""
        system = ChemicalSystem(atoms=[
            Atom(symbol="C", coords=np.array([0.0, 0.0, 0.0])),
            Atom(symbol="C", coords=np.array([1.5, 0.0, 0.0])),  # Bonded
            Atom(symbol="C", coords=np.array([5.0, 0.0, 0.0])),  # Not bonded
        ])
        selection = AtomSelection({0})

        partitioner = QMMMPartitioner(qm_selection=selection, buffer_radius=0)

        # Don't provide bonds - should auto-detect
        result = partitioner.partition(system)

        # Should find bond between 0 and 1
        assert (0, 1) in result.cut_bonds
