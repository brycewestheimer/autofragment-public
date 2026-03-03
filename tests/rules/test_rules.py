# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive tests for the fragmentation rules engine.

This module tests all the rules implementation including:
- RuleAction enum
- FragmentationRule ABC
- RuleEngine
- All common, biological, and materials science rules
"""
import pytest

from autofragment.core.types import Atom, ChemicalSystem
from autofragment.rules import (
    AlphaBetaCarbonRule,
    # Common
    AromaticRingRule,
    DisulfideBondRule,
    DoubleBondRule,
    FragmentationRule,
    FunctionalGroupRule,
    HydrogenBondRule,
    MetalCoordinationRule,
    MetalNodeRule,
    MOFLinkerRule,
    # Biological
    PeptideBondRule,
    PerovskiteOctahedralRule,
    ProlineRingRule,
    # Core
    RuleAction,
    RuleEngine,
    # Materials
    SiloxaneBridgeRule,
)


class TestRuleAction:
    """Tests for the RuleAction enum."""

    def test_all_actions_exist(self):
        """Test that all expected RuleAction values exist."""
        assert hasattr(RuleAction, "MUST_NOT_BREAK")
        assert hasattr(RuleAction, "PREFER_KEEP")
        assert hasattr(RuleAction, "ALLOW")
        assert hasattr(RuleAction, "PREFER_BREAK")

    def test_action_count(self):
        """Test that there are exactly 4 actions."""
        assert len(RuleAction) == 4


class TestFragmentationRule:
    """Tests for the FragmentationRule ABC."""

    def test_cannot_instantiate_abc(self):
        """Test that FragmentationRule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FragmentationRule()

    def test_priority_constants(self):
        """Test that priority constants are defined correctly."""
        assert FragmentationRule.PRIORITY_CRITICAL == 1000
        assert FragmentationRule.PRIORITY_HIGH == 800
        assert FragmentationRule.PRIORITY_MEDIUM == 500
        assert FragmentationRule.PRIORITY_LOW == 200
        assert FragmentationRule.PRIORITY_DEFAULT == 500

    def test_concrete_rule_instantiation(self):
        """Test that concrete rules can be instantiated."""
        rule = AromaticRingRule()
        assert isinstance(rule, FragmentationRule)
        assert rule.priority == FragmentationRule.PRIORITY_CRITICAL

    def test_custom_priority(self):
        """Test that rules accept custom priority."""
        rule = AromaticRingRule(priority=999)
        assert rule.priority == 999

    def test_priority_setter(self):
        """Test that priority can be modified after creation."""
        rule = AromaticRingRule()
        rule.priority = 123
        assert rule.priority == 123


class TestRuleEngine:
    """Tests for the RuleEngine class."""

    def test_empty_engine(self):
        """Test empty engine initialization."""
        engine = RuleEngine()
        assert len(engine) == 0
        assert engine.get_rules() == []

    def test_engine_with_rules(self):
        """Test engine initialization with rules."""
        rules = [AromaticRingRule(), DoubleBondRule()]
        engine = RuleEngine(rules)
        assert len(engine) == 2

    def test_add_rule(self):
        """Test adding rules to engine."""
        engine = RuleEngine()
        engine.add_rule(AromaticRingRule())
        assert len(engine) == 1

    def test_priority_ordering(self):
        """Test that rules are maintained in priority order."""
        low_priority = HydrogenBondRule()  # PRIORITY_LOW
        high_priority = AromaticRingRule()  # PRIORITY_CRITICAL

        engine = RuleEngine([low_priority, high_priority])
        rules = engine.get_rules()

        # Higher priority should come first
        assert rules[0].priority >= rules[1].priority

    def test_remove_rule(self):
        """Test removing a rule by name."""
        engine = RuleEngine([AromaticRingRule(), DoubleBondRule()])
        assert len(engine) == 2

        removed = engine.remove_rule("aromatic_ring")
        assert removed is True
        assert len(engine) == 1

        # Removing non-existent rule returns False
        removed = engine.remove_rule("nonexistent")
        assert removed is False

    def test_clear_rules(self):
        """Test clearing all rules."""
        engine = RuleEngine([AromaticRingRule(), DoubleBondRule()])
        engine.clear_rules()
        assert len(engine) == 0

    def test_conflict_resolution_empty(self):
        """Test conflict resolution with no applicable rules."""
        engine = RuleEngine()
        result = engine.resolve_conflict([])
        assert result == RuleAction.ALLOW

    def test_conflict_resolution_most_restrictive(self):
        """Test that most restrictive action wins."""
        engine = RuleEngine()

        # Create mock rule/action pairs
        rule1 = AromaticRingRule()
        rule2 = HydrogenBondRule()

        actions = [
            (rule2, RuleAction.ALLOW),
            (rule1, RuleAction.MUST_NOT_BREAK),
        ]

        result = engine.resolve_conflict(actions)
        assert result == RuleAction.MUST_NOT_BREAK

    def test_repr(self):
        """Test string representation."""
        engine = RuleEngine([AromaticRingRule()])
        repr_str = repr(engine)
        assert "RuleEngine" in repr_str
        assert "aromatic_ring" in repr_str


class TestCommonRules:
    """Tests for common chemical rules."""

    def test_aromatic_ring_rule_defaults(self):
        """Test AromaticRingRule default values."""
        rule = AromaticRingRule()
        assert rule.name == "aromatic_ring"
        assert rule.priority == FragmentationRule.PRIORITY_CRITICAL
        assert rule.action() == RuleAction.MUST_NOT_BREAK

    def test_double_bond_rule_defaults(self):
        """Test DoubleBondRule default values."""
        rule = DoubleBondRule()
        assert rule.name == "double_bond"
        assert rule.min_order == 1.5
        assert rule.action() == RuleAction.MUST_NOT_BREAK

    def test_double_bond_rule_custom_order(self):
        """Test DoubleBondRule with custom min_order."""
        rule = DoubleBondRule(min_order=2.0)
        assert rule.min_order == 2.0

    def test_metal_coordination_rule_defaults(self):
        """Test MetalCoordinationRule default values."""
        rule = MetalCoordinationRule()
        assert rule.name == "metal_coordination"
        assert rule.priority == FragmentationRule.PRIORITY_HIGH
        assert "Fe" in rule.metals
        assert "Cu" in rule.metals

    def test_metal_coordination_rule_custom_metals(self):
        """Test MetalCoordinationRule with custom metal set."""
        rule = MetalCoordinationRule(metals={"Pt", "Pd"})
        assert rule.metals == {"Pt", "Pd"}

    def test_functional_group_rule_defaults(self):
        """Test FunctionalGroupRule default values."""
        rule = FunctionalGroupRule()
        assert rule.name == "functional_group"
        assert "carboxyl" in rule.groups
        assert "nitro" in rule.groups


class TestBiologicalRules:
    """Tests for biological rules."""

    def test_peptide_bond_rule_configurable(self):
        """Test that PeptideBondRule is configurable."""
        # Default action
        rule1 = PeptideBondRule()
        assert rule1.action() == RuleAction.PREFER_KEEP

        # Custom action
        rule2 = PeptideBondRule(rule_action=RuleAction.PREFER_BREAK)
        assert rule2.action() == RuleAction.PREFER_BREAK

    def test_disulfide_bond_rule_configurable(self):
        """Test that DisulfideBondRule is configurable."""
        rule1 = DisulfideBondRule()
        assert rule1.action() == RuleAction.PREFER_KEEP

        rule2 = DisulfideBondRule(rule_action=RuleAction.MUST_NOT_BREAK)
        assert rule2.action() == RuleAction.MUST_NOT_BREAK

    def test_alpha_beta_carbon_rule_defaults(self):
        """Test AlphaBetaCarbonRule default values."""
        rule = AlphaBetaCarbonRule()
        assert rule.name == "alpha_beta_carbon"
        assert rule.action() == RuleAction.PREFER_BREAK

    def test_proline_ring_rule_defaults(self):
        """Test ProlineRingRule default values."""
        rule = ProlineRingRule()
        assert rule.name == "proline_ring"
        assert rule.priority == FragmentationRule.PRIORITY_CRITICAL
        assert rule.action() == RuleAction.MUST_NOT_BREAK

    def test_hydrogen_bond_rule_defaults_to_allow(self):
        """Test that HydrogenBondRule defaults to ALLOW (not restrictive)."""
        rule = HydrogenBondRule()
        assert rule.action() == RuleAction.ALLOW
        assert rule.priority == FragmentationRule.PRIORITY_LOW

    def test_hydrogen_bond_rule_geometry_params(self):
        """Test HydrogenBondRule geometry parameters."""
        rule = HydrogenBondRule(distance_cutoff=4.0, angle_cutoff=110.0)
        assert rule.distance_cutoff == 4.0
        assert rule.angle_cutoff == 110.0


class TestMaterialsScienceRules:
    """Tests for materials science rules."""

    def test_siloxane_bridge_rule_defaults(self):
        """Test SiloxaneBridgeRule default values."""
        rule = SiloxaneBridgeRule()
        assert rule.name == "siloxane_bridge"
        assert rule.action() == RuleAction.ALLOW

    def test_siloxane_bridge_rule_custom_action(self):
        """Test SiloxaneBridgeRule with custom action."""
        rule = SiloxaneBridgeRule(rule_action=RuleAction.PREFER_BREAK)
        assert rule.action() == RuleAction.PREFER_BREAK

    def test_mof_linker_rule_defaults(self):
        """Test MOFLinkerRule default values."""
        rule = MOFLinkerRule()
        assert rule.name == "mof_linker"
        assert rule.action() == RuleAction.MUST_NOT_BREAK
        assert "C" in rule.organic_elements
        assert "Zn" in rule.MOF_METALS

    def test_metal_node_rule_defaults(self):
        """Test MetalNodeRule default values."""
        rule = MetalNodeRule()
        assert rule.name == "metal_node"
        assert rule.priority == FragmentationRule.PRIORITY_CRITICAL
        assert rule.cluster_distance == 4.0

    def test_perovskite_rule_defaults(self):
        """Test PerovskiteOctahedralRule default values."""
        rule = PerovskiteOctahedralRule()
        assert rule.name == "perovskite_octahedral"
        assert "Ti" in rule.b_site_metals
        assert "Zr" in rule.b_site_metals

    def test_perovskite_rule_custom_metals(self):
        """Test PerovskiteOctahedralRule with custom B-site metals."""
        rule = PerovskiteOctahedralRule(b_site_metals={"Ti", "Nb"})
        assert rule.b_site_metals == {"Ti", "Nb"}


class TestRuleIntegration:
    """Integration tests for rules with ChemicalSystem."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple water molecule system."""
        atoms = [
            Atom(symbol="O", coords=(0.0, 0.0, 0.0)),
            Atom(symbol="H", coords=(0.96, 0.0, 0.0)),
            Atom(symbol="H", coords=(-0.23, 0.93, 0.0)),
        ]
        bonds = [
            {"atom1": 0, "atom2": 1, "order": 1.0},
            {"atom1": 0, "atom2": 2, "order": 1.0},
        ]
        return ChemicalSystem(atoms=atoms, bonds=bonds)

    @pytest.fixture
    def benzene_system(self):
        """Create a simplified benzene-like system for testing."""
        # Simplified benzene with aromatic bonds
        atoms = [
            Atom(symbol="C", coords=(0.0, 1.4, 0.0)),
            Atom(symbol="C", coords=(1.2, 0.7, 0.0)),
            Atom(symbol="C", coords=(1.2, -0.7, 0.0)),
            Atom(symbol="C", coords=(0.0, -1.4, 0.0)),
            Atom(symbol="C", coords=(-1.2, -0.7, 0.0)),
            Atom(symbol="C", coords=(-1.2, 0.7, 0.0)),
        ]
        bonds = [
            {"atom1": 0, "atom2": 1, "order": 1.5, "aromatic": True},
            {"atom1": 1, "atom2": 2, "order": 1.5, "aromatic": True},
            {"atom1": 2, "atom2": 3, "order": 1.5, "aromatic": True},
            {"atom1": 3, "atom2": 4, "order": 1.5, "aromatic": True},
            {"atom1": 4, "atom2": 5, "order": 1.5, "aromatic": True},
            {"atom1": 5, "atom2": 0, "order": 1.5, "aromatic": True},
        ]
        return ChemicalSystem(atoms=atoms, bonds=bonds)

    def test_engine_evaluate_bond(self, simple_system):
        """Test evaluating a bond with empty engine."""
        engine = RuleEngine()
        action = engine.evaluate_bond((0, 1), simple_system)
        # No rules, should return ALLOW
        assert action == RuleAction.ALLOW

    def test_engine_get_bond_actions(self, simple_system):
        """Test getting all bond actions."""
        engine = RuleEngine()
        actions = engine.get_bond_actions(simple_system)
        assert len(actions) == 2  # Two O-H bonds
        for bond, action in actions.items():
            assert action == RuleAction.ALLOW

    def test_engine_get_breakable_bonds(self, simple_system):
        """Test getting breakable bonds."""
        engine = RuleEngine()
        breakable = engine.get_breakable_bonds(simple_system)
        assert len(breakable) == 2  # Both O-H bonds are breakable

    def test_aromatic_rule_with_benzene(self, benzene_system):
        """Test AromaticRingRule with benzene system."""
        rule = AromaticRingRule()
        engine = RuleEngine([rule])

        # All benzene C-C bonds should be protected
        for bond_info in benzene_system.bonds:
            bond = (bond_info["atom1"], bond_info["atom2"])
            action = engine.evaluate_bond(bond, benzene_system)
            assert action == RuleAction.MUST_NOT_BREAK

    def test_double_bond_rule_applies(self):
        """Test DoubleBondRule with a double bond."""
        atoms = [
            Atom(symbol="C", coords=(0.0, 0.0, 0.0)),
            Atom(symbol="O", coords=(1.2, 0.0, 0.0)),
        ]
        bonds = [{"atom1": 0, "atom2": 1, "order": 2.0}]
        system = ChemicalSystem(atoms=atoms, bonds=bonds)

        rule = DoubleBondRule()
        assert rule.applies_to((0, 1), system) is True
        assert rule.action() == RuleAction.MUST_NOT_BREAK

    def test_all_rules_in_engine(self):
        """Test loading all rules into a single engine."""
        all_rules = [
            AromaticRingRule(),
            DoubleBondRule(),
            MetalCoordinationRule(),
            FunctionalGroupRule(),
            PeptideBondRule(),
            DisulfideBondRule(),
            AlphaBetaCarbonRule(),
            ProlineRingRule(),
            HydrogenBondRule(),
            SiloxaneBridgeRule(),
            MOFLinkerRule(),
            MetalNodeRule(),
            PerovskiteOctahedralRule(),
        ]

        engine = RuleEngine(all_rules)
        assert len(engine) == 13

        # Verify priority ordering
        rules = engine.get_rules()
        for i in range(len(rules) - 1):
            assert rules[i].priority >= rules[i + 1].priority
