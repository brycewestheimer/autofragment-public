# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-level layer definitions."""

import numpy as np
import pytest

from autofragment.multilevel import (
    ComputationalLayer,
    EmbeddingType,
    LayerType,
    LinkAtom,
    MultiLevelScheme,
    ONIOMScheme,
    assign_by_custom,
    assign_by_distance,
    assign_by_element,
    assign_by_residue,
    assign_by_residue_number,
    create_oniom_scheme,
    expand_selection_to_residues,
    parse_method_basis,
    validate_layer_assignment,
)
from autofragment.multilevel.link_atoms import (
    LinkAtomInfo,
    calculate_g_factor,
    create_link_atoms_for_cut_bonds,
    get_bond_length,
    position_link_atom_fixed_distance,
    position_link_atom_gfactor,
    validate_link_atoms,
)
from autofragment.multilevel.point_charges import (
    PointCharge,
    PointChargeEmbedding,
    generate_simple_charge_array,
)


class TestLayerType:
    """Tests for LayerType enum."""

    def test_layer_type_values(self):
        """Test that all layer types have expected values."""
        assert LayerType.HIGH.value == "high"
        assert LayerType.MEDIUM.value == "medium"
        assert LayerType.LOW.value == "low"
        assert LayerType.MM.value == "mm"

    def test_layer_type_from_string(self):
        """Test creating LayerType from string value."""
        assert LayerType("high") == LayerType.HIGH
        assert LayerType("mm") == LayerType.MM


class TestLinkAtom:
    """Tests for LinkAtom dataclass."""

    def test_link_atom_creation(self):
        """Test basic link atom creation."""
        link = LinkAtom(qm_atom_index=0, mm_atom_index=5)
        assert link.qm_atom_index == 0
        assert link.mm_atom_index == 5
        assert link.element == "H"
        assert link.position is None
        assert link.scale_factor == 0.723

    def test_link_atom_with_position(self):
        """Test link atom with position."""
        pos = np.array([1.0, 2.0, 3.0])
        link = LinkAtom(qm_atom_index=0, mm_atom_index=1, position=pos)
        assert link.position is not None
        np.testing.assert_array_equal(link.position, pos)

    def test_link_atom_invalid_position(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError, match="3D vector"):
            LinkAtom(qm_atom_index=0, mm_atom_index=1, position=np.array([1.0, 2.0]))

    def test_link_atom_to_dict(self):
        """Test serialization to dictionary."""
        link = LinkAtom(
            qm_atom_index=0,
            mm_atom_index=5,
            element="H",
            position=np.array([1.0, 2.0, 3.0]),
            scale_factor=0.8,
        )
        d = link.to_dict()
        assert d["qm_atom_index"] == 0
        assert d["mm_atom_index"] == 5
        assert d["element"] == "H"
        assert d["position"] == [1.0, 2.0, 3.0]
        assert d["scale_factor"] == 0.8

    def test_link_atom_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "qm_atom_index": 2,
            "mm_atom_index": 10,
            "element": "H",
            "position": [4.0, 5.0, 6.0],
            "scale_factor": 0.75,
        }
        link = LinkAtom.from_dict(d)
        assert link.qm_atom_index == 2
        assert link.mm_atom_index == 10
        np.testing.assert_array_equal(link.position, [4.0, 5.0, 6.0])


class TestComputationalLayer:
    """Tests for ComputationalLayer dataclass."""

    def test_qm_layer_creation(self):
        """Test creating a QM layer."""
        layer = ComputationalLayer(
            name="active_site",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="6-31G*",
            atom_indices={0, 1, 2, 3, 4},
        )
        assert layer.name == "active_site"
        assert layer.layer_type == LayerType.HIGH
        assert layer.method == "B3LYP"
        assert layer.basis_set == "6-31G*"
        assert layer.n_atoms == 5
        assert layer.is_qm is True

    def test_mm_layer_creation(self):
        """Test creating an MM layer (no basis set required)."""
        layer = ComputationalLayer(
            name="solvent",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices=set(range(100, 200)),
        )
        assert layer.name == "solvent"
        assert layer.layer_type == LayerType.MM
        assert layer.basis_set is None
        assert layer.is_qm is False
        assert layer.n_atoms == 100

    def test_qm_layer_requires_basis(self):
        """Test that QM layers require a basis set."""
        with pytest.raises(ValueError, match="requires a basis set"):
            ComputationalLayer(
                name="test",
                layer_type=LayerType.HIGH,
                method="B3LYP",
                # No basis_set provided
            )

    def test_medium_layer_requires_basis(self):
        """Test that MEDIUM layers also require basis set."""
        with pytest.raises(ValueError, match="requires a basis set"):
            ComputationalLayer(
                name="test",
                layer_type=LayerType.MEDIUM,
                method="HF",
            )

    def test_add_atoms(self):
        """Test adding atoms to a layer."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
        )
        assert layer.n_atoms == 0
        layer.add_atoms([1, 2, 3])
        assert layer.n_atoms == 3
        layer.add_atoms([4, 5])
        assert layer.n_atoms == 5
        assert layer.atom_indices == {1, 2, 3, 4, 5}

    def test_remove_atoms(self):
        """Test removing atoms from a layer."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3, 4, 5},
        )
        layer.remove_atoms([2, 4])
        assert layer.atom_indices == {1, 3, 5}

    def test_remove_atoms_also_unfreezes(self):
        """Test that removing atoms also removes from frozen set."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3, 4, 5},
            frozen_atoms={2, 3},
        )
        layer.remove_atoms([2])
        assert layer.atom_indices == {1, 3, 4, 5}
        assert layer.frozen_atoms == {3}

    def test_frozen_atoms_validation(self):
        """Test that frozen atoms must be in layer."""
        with pytest.raises(ValueError, match="not in layer"):
            ComputationalLayer(
                name="test",
                layer_type=LayerType.MM,
                method="AMBER",
                atom_indices={1, 2, 3},
                frozen_atoms={4, 5},  # Not in atom_indices
            )

    def test_freeze_atoms(self):
        """Test freezing atoms."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3, 4, 5},
        )
        layer.freeze_atoms([1, 2])
        assert layer.frozen_atoms == {1, 2}

    def test_freeze_invalid_atoms(self):
        """Test freezing atoms not in layer raises error."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3},
        )
        with pytest.raises(ValueError, match="Cannot freeze"):
            layer.freeze_atoms([1, 10])

    def test_unfreeze_atoms(self):
        """Test unfreezing specific atoms."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3},
            frozen_atoms={1, 2, 3},
        )
        layer.unfreeze_atoms([1, 2])
        assert layer.frozen_atoms == {3}

    def test_unfreeze_all(self):
        """Test unfreezing all atoms."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.MM,
            method="AMBER",
            atom_indices={1, 2, 3},
            frozen_atoms={1, 2, 3},
        )
        layer.unfreeze_atoms()
        assert layer.frozen_atoms == set()

    def test_add_link_atom(self):
        """Test adding link atoms."""
        layer = ComputationalLayer(
            name="qm",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="6-31G*",
            atom_indices={0, 1, 2},
        )
        layer.add_link_atom(qm_atom_index=2, mm_atom_index=3)
        assert len(layer.link_atoms) == 1
        assert layer.link_atoms[0].qm_atom_index == 2
        assert layer.link_atoms[0].mm_atom_index == 3

    def test_total_atoms_with_links(self):
        """Test total_atoms includes link atoms."""
        layer = ComputationalLayer(
            name="qm",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="6-31G*",
            atom_indices={0, 1, 2},
        )
        layer.add_link_atom(qm_atom_index=2, mm_atom_index=3)
        layer.add_link_atom(qm_atom_index=1, mm_atom_index=4)
        assert layer.n_atoms == 3
        assert layer.total_atoms == 5

    def test_get_boundary_atoms(self):
        """Test getting boundary atoms."""
        layer = ComputationalLayer(
            name="qm",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="6-31G*",
            atom_indices={0, 1, 2},
        )
        layer.add_link_atom(qm_atom_index=1, mm_atom_index=3)
        layer.add_link_atom(qm_atom_index=2, mm_atom_index=4)
        assert layer.get_boundary_atoms() == {1, 2}

    def test_charge_and_multiplicity(self):
        """Test charge and multiplicity attributes."""
        layer = ComputationalLayer(
            name="radical",
            layer_type=LayerType.HIGH,
            method="UB3LYP",
            basis_set="6-31G*",
            charge=-1,
            multiplicity=2,
        )
        assert layer.charge == -1
        assert layer.multiplicity == 2

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        layer = ComputationalLayer(
            name="test",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="cc-pVDZ",
            atom_indices={0, 1, 2},
            charge=1,
            multiplicity=2,
            frozen_atoms={0},
        )
        layer.add_link_atom(qm_atom_index=2, mm_atom_index=5)

        d = layer.to_dict()
        rebuilt = ComputationalLayer.from_dict(d)

        assert rebuilt.name == layer.name
        assert rebuilt.layer_type == layer.layer_type
        assert rebuilt.method == layer.method
        assert rebuilt.basis_set == layer.basis_set
        assert rebuilt.atom_indices == layer.atom_indices
        assert rebuilt.charge == layer.charge
        assert rebuilt.multiplicity == layer.multiplicity
        assert rebuilt.frozen_atoms == layer.frozen_atoms
        assert len(rebuilt.link_atoms) == 1

    def test_repr(self):
        """Test string representation."""
        layer = ComputationalLayer(
            name="active",
            layer_type=LayerType.HIGH,
            method="B3LYP",
            basis_set="6-31G*",
            atom_indices={0, 1, 2},
        )
        r = repr(layer)
        assert "active" in r
        assert "high" in r
        assert "B3LYP" in r
        assert "3" in r  # n_atoms


class TestMultiLevelScheme:
    """Tests for MultiLevelScheme container."""

    def create_two_layer_scheme(self) -> MultiLevelScheme:
        """Helper to create a standard two-layer scheme."""
        scheme = MultiLevelScheme(name="test_scheme")
        scheme.add_layer(
            ComputationalLayer(
                name="qm",
                layer_type=LayerType.HIGH,
                method="B3LYP",
                basis_set="6-31G*",
                atom_indices={0, 1, 2, 3, 4},
            )
        )
        scheme.add_layer(
            ComputationalLayer(
                name="mm",
                layer_type=LayerType.MM,
                method="AMBER",
                atom_indices=set(range(5, 100)),
            )
        )
        return scheme

    def test_scheme_creation(self):
        """Test creating a multi-level scheme."""
        scheme = MultiLevelScheme(name="ONIOM")
        assert scheme.name == "ONIOM"
        assert scheme.n_layers == 0

    def test_add_layer(self):
        """Test adding layers to scheme."""
        scheme = self.create_two_layer_scheme()
        assert scheme.n_layers == 2

    def test_get_layer_by_name(self):
        """Test getting layer by name."""
        scheme = self.create_two_layer_scheme()
        qm = scheme.get_layer("qm")
        assert qm is not None
        assert qm.method == "B3LYP"

        missing = scheme.get_layer("nonexistent")
        assert missing is None

    def test_get_layer_by_type(self):
        """Test getting layers by type."""
        scheme = self.create_two_layer_scheme()
        high_layers = scheme.get_layer_by_type(LayerType.HIGH)
        assert len(high_layers) == 1
        assert high_layers[0].name == "qm"

        mm_layers = scheme.get_layer_by_type(LayerType.MM)
        assert len(mm_layers) == 1
        assert mm_layers[0].name == "mm"

    def test_get_atom_layer(self):
        """Test finding which layer contains an atom."""
        scheme = self.create_two_layer_scheme()
        assert scheme.get_atom_layer(0).name == "qm"
        assert scheme.get_atom_layer(50).name == "mm"
        assert scheme.get_atom_layer(1000) is None

    def test_get_all_atoms(self):
        """Test getting all atoms from all layers."""
        scheme = self.create_two_layer_scheme()
        all_atoms = scheme.get_all_atoms()
        assert len(all_atoms) == 100  # 5 QM + 95 MM

    def test_validate_valid_scheme(self):
        """Test validation of valid scheme."""
        scheme = self.create_two_layer_scheme()
        is_valid, errors = scheme.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_empty_scheme(self):
        """Test validation catches empty scheme."""
        scheme = MultiLevelScheme(name="empty")
        is_valid, errors = scheme.validate()
        assert is_valid is False
        assert "no layers" in errors[0]

    def test_validate_overlapping_atoms(self):
        """Test validation catches overlapping atoms."""
        scheme = MultiLevelScheme(name="bad")
        scheme.add_layer(
            ComputationalLayer(
                name="layer1",
                layer_type=LayerType.HIGH,
                method="B3LYP",
                basis_set="6-31G*",
                atom_indices={0, 1, 2},
            )
        )
        scheme.add_layer(
            ComputationalLayer(
                name="layer2",
                layer_type=LayerType.MM,
                method="AMBER",
                atom_indices={2, 3, 4},  # Overlaps at atom 2
            )
        )
        is_valid, errors = scheme.validate()
        assert is_valid is False
        assert any("multiple layers" in e for e in errors)

    def test_validate_empty_layer(self):
        """Test validation catches empty layers."""
        scheme = MultiLevelScheme(name="bad")
        scheme.add_layer(
            ComputationalLayer(
                name="empty_layer",
                layer_type=LayerType.MM,
                method="AMBER",
                # No atoms
            )
        )
        is_valid, errors = scheme.validate()
        assert is_valid is False
        assert any("no atoms" in e for e in errors)

    def test_get_oniom_string(self):
        """Test ONIOM method string generation."""
        scheme = self.create_two_layer_scheme()
        oniom_str = scheme.get_oniom_string()
        assert oniom_str == "ONIOM(B3LYP/6-31G*:AMBER)"

    def test_get_oniom_string_three_layer(self):
        """Test ONIOM string with three layers."""
        scheme = MultiLevelScheme(name="three_layer")
        scheme.add_layer(
            ComputationalLayer(
                name="high",
                layer_type=LayerType.HIGH,
                method="CCSD(T)",
                basis_set="cc-pVTZ",
                atom_indices={0, 1},
            )
        )
        scheme.add_layer(
            ComputationalLayer(
                name="medium",
                layer_type=LayerType.MEDIUM,
                method="B3LYP",
                basis_set="6-31G*",
                atom_indices={2, 3, 4},
            )
        )
        scheme.add_layer(
            ComputationalLayer(
                name="low",
                layer_type=LayerType.MM,
                method="UFF",
                atom_indices={5, 6, 7},
            )
        )
        oniom_str = scheme.get_oniom_string()
        assert oniom_str == "ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G*:UFF)"

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        scheme = self.create_two_layer_scheme()
        scheme.description = "Test scheme"

        d = scheme.to_dict()
        rebuilt = MultiLevelScheme.from_dict(d)

        assert rebuilt.name == scheme.name
        assert rebuilt.description == scheme.description
        assert rebuilt.n_layers == scheme.n_layers
        assert rebuilt.layers[0].name == "qm"
        assert rebuilt.layers[1].name == "mm"

    def test_repr(self):
        """Test string representation."""
        scheme = self.create_two_layer_scheme()
        r = repr(scheme)
        assert "test_scheme" in r
        assert "qm(5)" in r
        assert "mm(95)" in r

    def test_scheme_type_default(self):
        """Test default scheme type is oniom."""
        scheme = MultiLevelScheme(name="test")
        assert scheme.scheme_type == "oniom"

    def test_scheme_type_custom(self):
        """Test setting custom scheme type."""
        scheme = MultiLevelScheme(name="test", scheme_type="qmmm")
        assert scheme.scheme_type == "qmmm"

    def test_embedding_type_default(self):
        """Test default embedding is electrostatic."""
        scheme = MultiLevelScheme(name="test")
        assert scheme.embedding_type == EmbeddingType.ELECTROSTATIC
        assert scheme.electrostatic_embedding is True
        assert scheme.mechanical_embedding is False

    def test_embedding_type_mechanical(self):
        """Test mechanical embedding."""
        scheme = MultiLevelScheme(name="test", embedding_type=EmbeddingType.MECHANICAL)
        assert scheme.electrostatic_embedding is False
        assert scheme.mechanical_embedding is True

    def test_embedding_type_polarizable(self):
        """Test polarizable embedding."""
        scheme = MultiLevelScheme(name="test", embedding_type=EmbeddingType.POLARIZABLE)
        assert scheme.electrostatic_embedding is True  # includes electrostatic
        assert scheme.mechanical_embedding is False

    def test_total_atoms_property(self):
        """Test total_atoms property."""
        scheme = self.create_two_layer_scheme()
        assert scheme.total_atoms == 100  # 5 + 95

    def test_get_high_layer(self):
        """Test getting the high layer."""
        scheme = self.create_two_layer_scheme()
        high = scheme.get_high_layer()
        assert high is not None
        assert high.name == "qm"
        assert high.layer_type == LayerType.HIGH

    def test_get_high_layer_none(self):
        """Test get_high_layer returns None when no HIGH layer."""
        scheme = MultiLevelScheme(name="mm_only")
        scheme.add_layer(
            ComputationalLayer(
                name="mm",
                layer_type=LayerType.MM,
                method="AMBER",
                atom_indices={0, 1, 2},
            )
        )
        assert scheme.get_high_layer() is None

    def test_to_dict_includes_scheme_config(self):
        """Test serialization includes scheme_type and embedding."""
        scheme = MultiLevelScheme(
            name="test",
            scheme_type="qmmm",
            embedding_type=EmbeddingType.MECHANICAL,
        )
        d = scheme.to_dict()
        assert d["scheme_type"] == "qmmm"
        assert d["embedding_type"] == "mechanical"

    def test_from_dict_includes_scheme_config(self):
        """Test deserialization with scheme_type and embedding."""
        d = {
            "name": "test",
            "scheme_type": "subtractive",
            "embedding_type": "polarizable",
            "layers": [],
        }
        scheme = MultiLevelScheme.from_dict(d)
        assert scheme.scheme_type == "subtractive"
        assert scheme.embedding_type == EmbeddingType.POLARIZABLE


class TestEmbeddingType:
    """Tests for EmbeddingType enum."""

    def test_embedding_type_values(self):
        """Test embedding type values."""
        assert EmbeddingType.MECHANICAL.value == "mechanical"
        assert EmbeddingType.ELECTROSTATIC.value == "electrostatic"
        assert EmbeddingType.POLARIZABLE.value == "polarizable"

    def test_embedding_type_from_string(self):
        """Test creating from string."""
        assert EmbeddingType("mechanical") == EmbeddingType.MECHANICAL
        assert EmbeddingType("electrostatic") == EmbeddingType.ELECTROSTATIC


class TestAssignByDistance:
    """Tests for distance-based layer assignment."""

    def test_basic_distance_assignment(self):
        """Test basic distance-based assignment."""
        # 4 atoms in a line: 0-0-0, 2-0-0, 5-0-0, 10-0-0
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])

        layers = assign_by_distance(coords, center_atoms={0}, layer_cutoffs=[3.0, 7.0])

        assert len(layers) == 3  # center + 2 cutoff zones
        assert layers[0] == {0}  # center
        assert layers[1] == {1}  # within 3A
        assert layers[2] == {2, 3}  # within 7A (atom 2) and beyond (atom 3 grouped)

    def test_single_cutoff(self):
        """Test with single cutoff (two-layer)."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])

        layers = assign_by_distance(coords, center_atoms={0}, layer_cutoffs=[3.0])

        assert len(layers) == 2
        assert layers[0] == {0}
        assert layers[1] == {1, 2}

    def test_multiple_center_atoms(self):
        """Test with multiple center atoms."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],  # 5A from center 0, but 1A from center if 5 was center
        ])

        layers = assign_by_distance(
            coords, center_atoms={0, 2}, layer_cutoffs=[2.0]
        )

        assert layers[0] == {0, 2}  # centers
        assert layers[1] == {1, 3}  # within 2A of a center


class TestAssignByResidue:
    """Tests for residue-based assignment."""

    def test_basic_residue_assignment(self):
        """Test basic residue assignment."""
        residues = ["ALA", "ALA", "GLY", "WAT", "WAT"]
        qm, mm = assign_by_residue(residues, ["ALA", "GLY"])

        assert qm == {0, 1, 2}
        assert mm == {3, 4}

    def test_no_match(self):
        """Test when no residues match."""
        residues = ["WAT", "WAT", "WAT"]
        qm, mm = assign_by_residue(residues, ["ALA"])

        assert qm == set()
        assert mm == {0, 1, 2}

    def test_all_match(self):
        """Test when all residues match."""
        residues = ["HEM", "HEM"]
        qm, mm = assign_by_residue(residues, ["HEM"])

        assert qm == {0, 1}
        assert mm == set()


class TestAssignByResidueNumber:
    """Tests for residue number-based assignment."""

    def test_basic_resnum_assignment(self):
        """Test basic residue number assignment."""
        resnums = [1, 1, 1, 2, 2, 3, 3, 3]
        qm, mm = assign_by_residue_number(resnums, [1, 3])

        assert qm == {0, 1, 2, 5, 6, 7}
        assert mm == {3, 4}


class TestAssignByElement:
    """Tests for element-based assignment."""

    def test_basic_element_assignment(self):
        """Test basic element assignment."""
        elements = ["Fe", "N", "N", "N", "N", "C", "C"]
        qm, mm = assign_by_element(elements, ["Fe", "N"])

        assert qm == {0, 1, 2, 3, 4}
        assert mm == {5, 6}


class TestAssignByCustom:
    """Tests for custom selector assignment."""

    def test_basic_custom_assignment(self):
        """Test custom selector."""
        def selector(idx):
            if idx < 3:
                return "high"
            elif idx < 6:
                return "medium"
            else:
                return "low"

        layers = assign_by_custom(10, selector)

        assert layers["high"] == {0, 1, 2}
        assert layers["medium"] == {3, 4, 5}
        assert layers["low"] == {6, 7, 8, 9}


class TestExpandSelectionToResidues:
    """Tests for residue expansion."""

    def test_expand_to_residues(self):
        """Test expanding selection to complete residues."""
        # 3 residues, each with 3 atoms
        atom_to_residue = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        # Select just atom 1 (in residue 0)
        selection = {1}
        expanded = expand_selection_to_residues(selection, atom_to_residue)

        assert expanded == {0, 1, 2}  # All of residue 0

    def test_expand_multiple_residues(self):
        """Test expanding multiple residues."""
        atom_to_residue = [0, 0, 1, 1, 2, 2]
        selection = {1, 4}  # Atoms in residues 0 and 2
        expanded = expand_selection_to_residues(selection, atom_to_residue)

        assert expanded == {0, 1, 4, 5}  # Residues 0 and 2


class TestValidateLayerAssignment:
    """Tests for layer validation."""

    def test_valid_assignment(self):
        """Test valid layer assignment."""
        layers = [{0, 1}, {2, 3, 4}]
        is_valid, error = validate_layer_assignment(layers, 5)

        assert is_valid is True
        assert error is None

    def test_missing_atoms(self):
        """Test detection of missing atoms."""
        layers = [{0, 1}, {2, 3}]  # Missing atom 4
        is_valid, error = validate_layer_assignment(layers, 5)

        assert is_valid is False
        assert "not assigned" in error

    def test_overlapping_atoms(self):
        """Test detection of overlapping atoms."""
        layers = [{0, 1, 2}, {2, 3, 4}]  # Atom 2 in both
        is_valid, error = validate_layer_assignment(layers, 5)

        assert is_valid is False
        assert "multiple layers" in error


class TestParseMethodBasis:
    """Tests for method/basis parsing."""

    def test_qm_method_with_basis(self):
        """Test parsing QM method with basis."""
        result = parse_method_basis("B3LYP/6-31G*")
        assert result.method == "B3LYP"
        assert result.basis == "6-31G*"
        assert result.is_mm is False

    def test_mm_method(self):
        """Test parsing MM method."""
        result = parse_method_basis("AMBER")
        assert result.method == "AMBER"
        assert result.basis is None
        assert result.is_mm is True

    def test_uff_method(self):
        """Test parsing UFF."""
        result = parse_method_basis("UFF")
        assert result.is_mm is True


class TestONIOMScheme:
    """Tests for ONIOMScheme class."""

    def test_two_layer_creation(self):
        """Test creating a two-layer ONIOM scheme."""
        scheme = ONIOMScheme(
            high_method="B3LYP",
            high_basis="6-31G*",
            low_method="UFF",
        )

        assert scheme.n_layers == 2
        assert scheme.high_method == "B3LYP"
        assert scheme.high_basis == "6-31G*"
        assert scheme.low_method == "UFF"
        assert scheme.is_three_layer is False

    def test_three_layer_creation(self):
        """Test creating a three-layer ONIOM scheme."""
        scheme = ONIOMScheme(
            high_method="CCSD(T)",
            high_basis="cc-pVTZ",
            medium_method="B3LYP",
            medium_basis="6-31G*",
            low_method="AMBER",
        )

        assert scheme.n_layers == 3
        assert scheme.is_three_layer is True

    def test_from_string_two_layer(self):
        """Test parsing two-layer ONIOM string."""
        scheme = ONIOMScheme.from_string("ONIOM(B3LYP/6-31G*:UFF)")

        assert scheme.high_method == "B3LYP"
        assert scheme.high_basis == "6-31G*"
        assert scheme.low_method == "UFF"
        assert scheme.low_basis is None

    def test_from_string_three_layer(self):
        """Test parsing three-layer ONIOM string."""
        scheme = ONIOMScheme.from_string(
            "ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G*:AMBER)"
        )

        assert scheme.high_method == "CCSD(T)"
        assert scheme.high_basis == "cc-pVTZ"
        assert scheme.medium_method == "B3LYP"
        assert scheme.medium_basis == "6-31G*"
        assert scheme.low_method == "AMBER"

    def test_from_string_invalid(self):
        """Test error on invalid ONIOM string."""
        with pytest.raises(ValueError, match="Invalid ONIOM"):
            ONIOMScheme.from_string("not an oniom spec")

    def test_from_string_too_many_layers(self):
        """Test error on too many layers."""
        with pytest.raises(ValueError, match="2 or 3 layers"):
            ONIOMScheme.from_string("ONIOM(A:B:C:D)")

    def test_to_gaussian_input_two_layer(self):
        """Test Gaussian input for two-layer."""
        scheme = ONIOMScheme("B3LYP", "6-31G*", "UFF")
        gaussian = scheme.to_gaussian_input()

        assert gaussian == "ONIOM(B3LYP/6-31G*:UFF)"

    def test_to_gaussian_input_three_layer(self):
        """Test Gaussian input for three-layer."""
        scheme = ONIOMScheme(
            "CCSD(T)", "cc-pVTZ",
            low_method="AMBER",
            medium_method="B3LYP",
            medium_basis="6-31G*",
        )
        gaussian = scheme.to_gaussian_input()

        assert gaussian == "ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G*:AMBER)"

    def test_to_gamess_input(self):
        """Test GAMESS input generation."""
        scheme = ONIOMScheme("B3LYP", "6-31G*", "AMBER")
        gamess = scheme.to_gamess_input()

        assert gamess["qmmethod"] == "B3LYP"
        assert gamess["qmbasis"] == "6-31G*"
        assert gamess["mmmethod"] == "AMBER"
        assert gamess["embedding"] == "electrostatic"

    def test_to_gamess_simomm_string(self):
        """Test GAMESS SIMOMM string generation."""
        scheme = ONIOMScheme("B3LYP", "6-31G*", "AMBER")
        simomm = scheme.to_gamess_simomm_string()

        assert "$SIMOMM" in simomm
        assert "QMMM=.TRUE." in simomm
        assert "MM=AMBER" in simomm

    def test_set_layer_atoms(self):
        """Test setting layer atoms."""
        scheme = ONIOMScheme("B3LYP", "6-31G*", "UFF")
        scheme.set_layer_atoms("high", {0, 1, 2})
        scheme.set_layer_atoms("low", {3, 4, 5})

        assert scheme.get_layer_atoms("high") == {0, 1, 2}
        assert scheme.get_layer_atoms("low") == {3, 4, 5}

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        scheme = ONIOMScheme("B3LYP", "6-31G*", "UFF")
        scheme.set_layer_atoms("high", {0, 1, 2})

        d = scheme.to_dict()
        rebuilt = ONIOMScheme.from_dict(d)

        assert rebuilt.high_method == "B3LYP"
        assert rebuilt.high_basis == "6-31G*"
        assert rebuilt.low_method == "UFF"


class TestCreateOniomScheme:
    """Tests for create_oniom_scheme convenience function."""

    def test_create_with_explicit_atoms(self):
        """Test creating scheme with explicit atom sets."""
        scheme = create_oniom_scheme(
            high_method="B3LYP",
            high_basis="6-31G*",
            high_atoms={0, 1, 2},
            low_method="UFF",
            low_atoms={3, 4, 5, 6, 7},
        )

        assert scheme.get_layer_atoms("high") == {0, 1, 2}
        assert scheme.get_layer_atoms("low") == {3, 4, 5, 6, 7}

    def test_create_with_total_atoms(self):
        """Test creating scheme with total_atoms to compute low."""
        scheme = create_oniom_scheme(
            high_method="B3LYP",
            high_basis="6-31G*",
            high_atoms={0, 1, 2},
            low_method="UFF",
            total_atoms=10,
        )

        assert scheme.get_layer_atoms("high") == {0, 1, 2}
        assert scheme.get_layer_atoms("low") == {3, 4, 5, 6, 7, 8, 9}


class TestLinkAtomPositioning:
    """Tests for advanced link atom positioning."""

    def test_get_bond_length(self):
        """Test bond length lookup."""
        assert get_bond_length("C", "H") == 1.09
        assert get_bond_length("H", "C") == 1.09  # Order independent
        assert get_bond_length("C", "C") == 1.54

    def test_calculate_g_factor(self):
        """Test g-factor calculation."""
        g = calculate_g_factor("C", "C", "H")
        assert abs(g - 1.09/1.54) < 0.001

    def test_position_link_atom_gfactor(self):
        """Test g-factor positioning."""
        qm_pos = np.array([0.0, 0.0, 0.0])
        mm_pos = np.array([1.54, 0.0, 0.0])
        g_factor = 0.709

        link_pos = position_link_atom_gfactor(qm_pos, mm_pos, g_factor)

        expected_x = 0.709 * 1.54
        assert abs(link_pos[0] - expected_x) < 0.001
        assert link_pos[1] == 0.0
        assert link_pos[2] == 0.0

    def test_position_link_atom_fixed_distance(self):
        """Test fixed distance positioning."""
        qm_pos = np.array([0.0, 0.0, 0.0])
        mm_pos = np.array([3.0, 0.0, 0.0])

        link_pos = position_link_atom_fixed_distance(qm_pos, mm_pos, distance=1.09)

        assert abs(link_pos[0] - 1.09) < 0.001

    def test_link_atom_info(self):
        """Test LinkAtomInfo dataclass."""
        info = LinkAtomInfo(
            qm_atom_index=0,
            mm_atom_index=1,
            element="H",
            g_factor=0.709,
        )

        qm_coords = np.array([0.0, 0.0, 0.0])
        mm_coords = np.array([1.54, 0.0, 0.0])

        pos = info.compute_position(qm_coords, mm_coords)

        assert info.position is not None
        assert np.allclose(pos, info.position)

    def test_create_link_atoms_for_cut_bonds(self):
        """Test batch link atom creation."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [3.08, 0.0, 0.0],
        ])
        elements = ["C", "C", "C"]
        cut_bonds = [(0, 1), (1, 2)]

        link_atoms = create_link_atoms_for_cut_bonds(
            coords, elements, cut_bonds, "H"
        )

        assert len(link_atoms) == 2
        assert all(la.position is not None for la in link_atoms)

    def test_validate_link_atoms(self):
        """Test link atom validation."""
        info = LinkAtomInfo(
            qm_atom_index=0,
            mm_atom_index=1,
            element="H",
            g_factor=0.709,
        )
        info.position = np.array([0.0, 0.0, 0.0])  # Too close to QM

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        warnings = validate_link_atoms([info], coords, min_distance=0.8)

        assert len(warnings) > 0


class TestPointCharge:
    """Tests for PointCharge dataclass."""

    def test_point_charge_creation(self):
        """Test creating a point charge."""
        pc = PointCharge(
            position=np.array([1.0, 2.0, 3.0]),
            charge=0.5,
            atom_index=0,
            element="C",
        )
        assert pc.charge == 0.5
        assert pc.element == "C"

    def test_distance_to(self):
        """Test distance calculation."""
        pc = PointCharge(position=np.array([0.0, 0.0, 0.0]), charge=0.5)
        other = np.array([3.0, 4.0, 0.0])

        dist = pc.distance_to(other)
        assert abs(dist - 5.0) < 0.001

    def test_to_dict_and_back(self):
        """Test serialization."""
        pc = PointCharge(
            position=np.array([1.0, 2.0, 3.0]),
            charge=0.5,
            atom_index=5,
            element="N",
        )

        d = pc.to_dict()
        rebuilt = PointCharge.from_dict(d)

        assert rebuilt.charge == pc.charge
        assert rebuilt.atom_index == pc.atom_index


class TestPointChargeEmbedding:
    """Tests for PointChargeEmbedding class."""

    def test_generate_charges(self):
        """Test generating point charges."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        elements = ["C", "H", "O"]
        mm_atoms = {0, 1, 2}

        embedding = PointChargeEmbedding()
        charges = embedding.generate_charges(coords, elements, mm_atoms)

        assert len(charges) == 3

    def test_exclude_link_atoms(self):
        """Test excluding link atom positions."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        elements = ["C", "C", "C"]
        mm_atoms = {0, 1, 2}
        link_mm_atoms = {0}  # Exclude atom 0

        embedding = PointChargeEmbedding(exclude_link_atoms=True)
        charges = embedding.generate_charges(
            coords, elements, mm_atoms, link_mm_atoms
        )

        assert len(charges) == 2
        assert all(pc.atom_index != 0 for pc in charges)

    def test_custom_charges(self):
        """Test using custom charges."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "C"]
        mm_atoms = {0, 1}

        embedding = PointChargeEmbedding()
        charges = embedding.generate_charges(
            coords, elements, mm_atoms,
            custom_charges={0: 0.123, 1: -0.456}
        )

        assert charges[0].charge == 0.123
        assert charges[1].charge == -0.456

    def test_to_arrays(self):
        """Test converting to numpy arrays."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "C"]
        mm_atoms = {0, 1}

        embedding = PointChargeEmbedding()
        charges = embedding.generate_charges(coords, elements, mm_atoms)
        positions, charge_vals = embedding.to_arrays(charges)

        assert positions.shape == (2, 3)
        assert charge_vals.shape == (2,)

    def test_to_gamess_format(self):
        """Test GAMESS format output."""
        embedding = PointChargeEmbedding()
        charges = [
            PointCharge(position=np.array([1.0, 2.0, 3.0]), charge=0.5),
        ]

        output = embedding.to_gamess_format(charges)

        assert "$EFRAG" in output
        assert "NCHARG=1" in output

    def test_to_gaussian_format(self):
        """Test Gaussian format output."""
        embedding = PointChargeEmbedding()
        charges = [
            PointCharge(position=np.array([1.0, 2.0, 3.0]), charge=0.5),
        ]

        output = embedding.to_gaussian_format(charges)

        assert "1.00000000" in output
        assert "0.50000000" in output


class TestGenerateSimpleChargeArray:
    """Tests for convenience function."""

    def test_generate_simple_charge_array(self):
        """Test the simple convenience function."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "C"]
        mm_atoms = {0, 1}

        positions, charges = generate_simple_charge_array(
            coords, elements, mm_atoms
        )

        assert positions.shape == (2, 3)
        assert charges.shape == (2,)

