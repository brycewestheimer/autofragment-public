# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for materials science fragmentation rules."""
from autofragment.core.types import Atom, ChemicalSystem
from autofragment.rules.base import RuleAction
from autofragment.rules.matsci import MOFLinkerRule, SilanolRule, SiloxaneBridgeRule


def create_silica_dimer():
    """Create H3Si-O-SiH2-OH dimer structure."""
    atoms = [
        Atom("Si", [0.0, 0.0, 0.0]),  # 0
        Atom("O", [1.6, 0.0, 0.0]),   # 1 (Bridge)
        Atom("Si", [3.2, 0.0, 0.0]),  # 2
        Atom("O", [3.2, 1.6, 0.0]),   # 3 (Terminal/Silanol)
        Atom("H", [3.2, 2.5, 0.0]),   # 4 (Silanol H)
        # Dummy H's to satisfy valency roughly (not used in logic mostly)
        Atom("H", [-1.0, 0.0, 0.0]),
        Atom("H", [0.0, 1.0, 0.0]),
        Atom("H", [0.0, -1.0, 0.0]),
        Atom("H", [4.2, 0.0, 0.0]),
        Atom("H", [3.2, -1.0, 0.0]),
    ]
    bonds = [
        {"atom1": 0, "atom2": 1, "order": 1.0}, # Si-O bridge
        {"atom1": 1, "atom2": 2, "order": 1.0}, # O-Si bridge
        {"atom1": 2, "atom2": 3, "order": 1.0}, # Si-O silanol
        {"atom1": 3, "atom2": 4, "order": 1.0}, # O-H silanol
    ]
    # Add H bonds to Si (indices 5-9)
    # ... ignoring specific connectivity for test as rules focus on Si-O

    return ChemicalSystem(atoms=atoms, bonds=bonds)

def test_siloxane_bridge_rule():
    sys = create_silica_dimer()
    rule = SiloxaneBridgeRule(rule_action=RuleAction.PREFER_BREAK)

    # Bridge bonds: (0, 1) and (1, 2)
    assert rule.applies_to((0, 1), sys)
    assert rule.applies_to((1, 2), sys)

    # Non-bridge bonds
    assert not rule.applies_to((2, 3), sys) # Si-O silanol (O connects to 1 Si and 1 H)

    assert rule.action() == RuleAction.PREFER_BREAK

def test_silanol_rule():
    sys = create_silica_dimer()
    rule = SilanolRule(rule_action=RuleAction.MUST_NOT_BREAK)

    # Bridge bonds
    assert not rule.applies_to((0, 1), sys)

    # Silanol Si-O bond (2, 3)
    assert rule.applies_to((2, 3), sys)

    # Silanol O-H bond (3, 4) -- SilanolRule only checks Si-O!
    # Common rules usually protect X-H bonds separately.
    assert not rule.applies_to((3, 4), sys)

    assert rule.action() == RuleAction.MUST_NOT_BREAK

def test_mof_linker_rule():
    # Benzene ring (organic) + Zn (metal)
    atoms = [
        Atom("C", [0,0,0]), Atom("C", [1,0,0]), # Organic
        Atom("Zn", [5,0,0]) # Metal
    ]
    bonds = [
        {"atom1": 0, "atom2": 1}, # C-C
        {"atom1": 1, "atom2": 2}  # C-Zn
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=bonds)

    rule = MOFLinkerRule(rule_action=RuleAction.MUST_NOT_BREAK)

    assert rule.applies_to((0, 1), sys) # C-C is linker
    assert not rule.applies_to((1, 2), sys) # C-Zn is not linker internal

    assert rule.action() == RuleAction.MUST_NOT_BREAK

def test_zeolite_acid_site_rule():
    from autofragment.rules.matsci import ZeoliteAcidSiteRule
    # Zeolite acid site: Al-O(H)-Si
    atoms = [
        Atom("Al", [0,0,0]), Atom("O", [1.6,0,0]), Atom("Si", [3.2,0,0]),
        Atom("H", [1.6,1.0,0]) # H on O
    ]
    bonds = [
        {"atom1": 0, "atom2": 1}, # Al-O
        {"atom1": 1, "atom2": 2}, # O-Si
        {"atom1": 1, "atom2": 3}  # O-H
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=bonds)

    rule = ZeoliteAcidSiteRule()

    assert rule.applies_to((0, 1), sys) # Al-O protected
    assert rule.applies_to((1, 2), sys) # Si-O protected

    # Normal Si-O-Si bridge (no H) shouldn't be protected by THIS rule
    atoms2 = [Atom("Si", [0,0,0]), Atom("O", [1.6,0,0]), Atom("Si", [3.2,0,0])]
    bonds2 = [{"atom1": 0, "atom2": 1}, {"atom1": 1, "atom2": 2}]
    sys2 = ChemicalSystem(atoms=atoms2, bonds=bonds2)
    assert not rule.applies_to((0, 1), sys2)

def test_metal_carboxylate_rule():
    from autofragment.rules.matsci import MetalCarboxylateRule
    # Zn-O-C(=O)-...
    atoms = [
        Atom("Zn", [0,0,0]), Atom("O", [2,0,0]), Atom("C", [3,0,0]), Atom("O", [3,1,0])
    ]
    bonds = [
        {"atom1": 0, "atom2": 1}, # Zn-O
        {"atom1": 1, "atom2": 2}, # O-C
        {"atom1": 2, "atom2": 3}  # C=O
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=bonds)

    rule = MetalCarboxylateRule(rule_action=RuleAction.PREFER_BREAK)

    assert rule.applies_to((0, 1), sys) # Zn-O breakable
    assert not rule.applies_to((1, 2), sys) # O-C not covered by this rule logic explicitly (only M-O)

def test_polymer_backbone_rule():
    from autofragment.rules.matsci import PolymerBackboneRule
    # C-C-C-C chain
    atoms = [
        Atom("C", [0,0,0]), Atom("C", [1.5,0,0]), Atom("C", [3.0,0,0]), Atom("C", [4.5,0,0])
    ]
    bonds = [
        {"atom1": 0, "atom2": 1}, # C1-C2
        {"atom1": 1, "atom2": 2}, # C2-C3
        {"atom1": 2, "atom2": 3}  # C3-C4
    ]
    sys = ChemicalSystem(atoms=atoms, bonds=bonds)

    rule = PolymerBackboneRule()

    # C1 (idx 0) has 1 C neighbor (C2)
    # C2 (idx 1) has 2 C neighbors (C1, C3)
    # C3 (idx 2) has 2 C neighbors (C2, C4)
    # C4 (idx 3) has 1 C neighbor (C3)

    # C2-C3 bond has C2(2 neighbors) and C3(2 neighbors).
    assert rule.applies_to((1, 2), sys)

    # C1-C2 bond has C1(1 neighbor). Should NOT apply (terminal?)
    assert not rule.applies_to((0, 1), sys)
