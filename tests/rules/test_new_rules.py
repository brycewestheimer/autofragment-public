# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

from autofragment.rules.base import BondRule, RuleAction, RuleEngine, RuleSet


class MockAtom:
    def __init__(self, symbol):
        self.symbol = symbol

class MockSystem:
    def __init__(self, atoms):
        self.atoms = [MockAtom(s) for s in atoms]
    def get_distance(self, i, j):
        return 1.5

def test_bond_rule():
    system = MockSystem(["C", "N"])
    rule = BondRule("CN", "C", "N", RuleAction.PREFER_KEEP)

    assert rule.applies_to((0, 1), system)
    assert not rule.applies_to((0, 0), system)
    assert rule.action() == RuleAction.PREFER_KEEP

def test_ruleset_and_engine():
    rs = RuleSet("test")
    rule = BondRule("CC", "C", "C", RuleAction.MUST_NOT_BREAK)
    rs.add(rule)

    assert len(rs.rules) == 1

    engine = RuleEngine(rs.rules)
    assert len(engine.get_rules()) == 1

    system = MockSystem(["C", "C"])
    assert engine.evaluate_bond((0, 1), system) == RuleAction.MUST_NOT_BREAK

def test_ruleset_from_rules():
    rules = [BondRule("C1", "C", "C"), BondRule("N1", "N", "N")]
    rs = RuleSet.from_rules(rules, "batch")
    assert len(rs.rules) == 2
    assert rs.name == "batch"
