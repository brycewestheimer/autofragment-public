# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for fragmentation scoring system."""
from autofragment.algorithms.scoring import (
    FragmentationScore,
    ScoringWeights,
    bond_breaking_penalty,
    chemical_integrity_score,
    computational_cost_estimate,
    interface_score,
    size_variance_penalty,
    total_bond_penalty,
)
from autofragment.rules.base import RuleAction


def test_initialization():
    """Test dataclass initialization with default and custom values."""
    score = FragmentationScore()
    assert score.bond_penalty == 0.0
    assert score.n_fragments == 0

    score = FragmentationScore(bond_penalty=5.0, size_variance=1.0)
    assert score.bond_penalty == 5.0
    assert score.size_variance == 1.0

def test_total_default_weights():
    """Test total score calculation with default weights."""
    score = FragmentationScore(
        bond_penalty=2.0,
        size_variance=0.5,
        interface_score=1.0,
        integrity_score=0.9,
        cost_estimate=10.0
    )
    # Total = -2.0 - 0.5 + 1.0 + 0.9 - 10.0 = -10.6
    assert abs(score.total() - (-10.6)) < 1e-9

def test_total_custom_weights():
    """Test total score calculation with custom weights."""
    score = FragmentationScore(
        bond_penalty=2.0,
        size_variance=0.5,
        interface_score=1.0
    )
    weights = {
        "bond_penalty": 2.0,
        "size_variance": 0.5,
        "interface_score": 10.0
    }
    # Total = -(2.0*2.0) - (0.5*0.5) + (1.0*10.0) = -4.0 - 0.25 + 10.0 = 5.75
    assert abs(score.total(weights) - 5.75) < 1e-9

def test_to_dict():
    """Test serialization to dictionary."""
    score = FragmentationScore(bond_penalty=1.0, fragment_sizes=[10, 20])
    data = score.to_dict()
    assert data["bond_penalty"] == 1.0
    assert data["fragment_sizes"] == [10, 20]
    assert data["n_fragments"] == 0

class MockGraph:
    def __init__(self):
        self.bonds = {}

    def get_bond(self, u, v):
        if (u, v) in self.bonds:
            return self.bonds[(u, v)]
        if (v, u) in self.bonds:
            return self.bonds[(v, u)]
        return None

    def add(self, u, v, order=1.0, bond_type=None):
        self.bonds[(u, v)] = {"order": order, "bond_type": bond_type}

    def find_rings(self):
        return self._rings

    def set_rings(self, rings):
        self._rings = rings

    _rings = []

class MockSystem:
    def __init__(self):
        self.graph = MockGraph()

class MockRuleEngine:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def evaluate_bond(self, bond, system):
        # check both directions
        if bond in self.overrides:
            return self.overrides[bond]
        if (bond[1], bond[0]) in self.overrides:
            return self.overrides[(bond[1], bond[0])]
        return RuleAction.ALLOW_BREAK

def test_bond_penalty_basic():
    system = MockSystem()
    system.graph.add(0, 1, order=1.0)
    system.graph.add(1, 2, order=2.0)

    assert bond_breaking_penalty((0, 1), system) == 1.0
    assert bond_breaking_penalty((1, 2), system) == 2.0
    # Non-existent
    assert bond_breaking_penalty((0, 2), system) == 0.0

def test_bond_penalty_aromatic():
    system = MockSystem()
    system.graph.add(0, 1, order=1.5, bond_type="aromatic")
    # 1.5 + 5.0 = 6.5
    assert bond_breaking_penalty((0, 1), system) == 6.5

def test_bond_penalty_rules():
    system = MockSystem()
    system.graph.add(0, 1, order=1.0)

    rule_engine = MockRuleEngine({(0, 1): RuleAction.MUST_NOT_BREAK})
    assert bond_breaking_penalty((0, 1), system, rule_engine) == float('inf')

    rule_engine = MockRuleEngine({(0, 1): RuleAction.PREFER_KEEP})
    assert bond_breaking_penalty((0, 1), system, rule_engine) == 2.0 # 1.0 * 2.0

    rule_engine = MockRuleEngine({(0, 1): RuleAction.PREFER_BREAK})
    assert bond_breaking_penalty((0, 1), system, rule_engine) == 0.5 # 1.0 * 0.5

def test_total_bond_penalty():
    system = MockSystem()
    system.graph.add(0, 1, order=1.0)
    system.graph.add(1, 2, order=2.0)

    broken = [(0, 1), (1, 2)]
    # 1.0 + 2.0 = 3.0
    assert total_bond_penalty(broken, system) == 3.0

def test_size_variance_penalty():
    # Equal sizes -> 0 penalty
    assert size_variance_penalty([10, 10, 10]) == 0.0

    # Unequal sizes
    # Mean = 20, Std = 8.16
    # CV = 8.16 / 20 = 0.408
    score = size_variance_penalty([10, 20, 30])
    assert score > 0.0
    assert abs(score - 0.408248) < 1e-4

    # Single fragment
    assert size_variance_penalty([100]) == 0.0

def test_interface_score():
    class SysWithCount:
        n_atoms = 10

    system = SysWithCount()

    # No break -> 1.0
    assert interface_score([], system) == 1.0

    # One break (2 atoms involved) -> 1 - 2/10 = 0.8
    assert interface_score([(0, 1)], system) == 0.8

    # Two breaks disjoint (4 atoms) -> 1 - 4/10 = 0.6
    assert interface_score([(0, 1), (2, 3)], system) == 0.6

    # Two breaks sharing atom (3 atoms) -> 1 - 3/10 = 0.7
    assert interface_score([(0, 1), (1, 2)], system) == 0.7

def test_chemical_integrity_score():
    system = MockSystem()
    # Ring 0-1-2
    system.graph.set_rings([[0, 1, 2]])

    # Partition preserving ring
    # F0: 0,1,2; F1: 3
    partition = [[0, 1, 2], [3]]
    assert chemical_integrity_score(partition, system) == 1.0

    # Partition splitting ring
    # F0: 0,1; F1: 2,3
    partition_split = [[0, 1], [2, 3]]
    # 0 preserved rings / 1 total
    assert chemical_integrity_score(partition_split, system) == 0.0

    # Two rings
    system.graph.set_rings([[0, 1, 2], [3, 4, 5]])
    # Split one, keep one
    partition_mixed = [[0, 1], [2, 3, 4, 5]] # Ring 1 split (0,1 vs 2), Ring 2 kept (3,4,5 in F1)
    assert chemical_integrity_score(partition_mixed, system) == 0.5

def test_computational_cost_estimate():
    # Single fragment = full cost = ratio 1.0
    assert computational_cost_estimate([10]) == 1.0

    # Linear scaling (alpha=1)
    # [5, 5] vs [10]: (5^1 + 5^1) / 10^1 = 10 / 10 = 1.0
    assert computational_cost_estimate([5, 5], scaling=1.0) == 1.0

    # Cubic scaling (alpha=3)
    # [5, 5] vs [10]: (125 + 125) / 1000 = 250 / 1000 = 0.25
    # Huge saving
    assert computational_cost_estimate([5, 5], scaling=3.0) == 0.25

    # Empty
    assert computational_cost_estimate([]) == 0.0

def test_scoring_weights():
    w = ScoringWeights()
    d = w.to_dict()
    assert d["bond_penalty"] == 1.0
    assert d["cost_estimate"] == 1.0

    w_fmo = ScoringWeights.for_fmo()
    assert w_fmo.bond_penalty == 2.0
    assert w_fmo.size_variance == 1.5

    w_mbe = ScoringWeights.for_mbe()
    assert w_mbe.size_variance == 2.0
