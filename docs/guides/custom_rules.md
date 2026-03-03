# Custom Rules Authoring Guide

This guide explains how to create custom fragmentation rules to control where bonds can be broken during molecular fragmentation.

## Rule System Overview

Rules encode chemical knowledge about fragmentation:
- **What to protect**: Aromatic rings, functional groups
- **Where to break**: Weak bonds, natural boundaries
- **Priority handling**: Conflict resolution between rules

## Rule Actions

Rules return one of four actions:

| Action | Description |
|--------|-------------|
| `MUST_NOT_BREAK` | Bond must never be broken |
| `PREFER_KEEP` | Prefer keeping, can break if necessary |
| `ALLOW` | No preference (neutral) |
| `PREFER_BREAK` | Good fragmentation point |

When multiple rules apply, **the most restrictive action wins**.

## Base Classes

`autofragment` provides two easy ways to create rules:
1. `BondRule`: A simple rule based on the element symbols of the atoms in a bond.
2. `FragmentationRule`: An abstract base class for creating rules with custom logic.

## Creating Bond Rules

Use `BondRule` to quickly protect or prefer specific atom-atom bonds:

```python
from autofragment.rules import BondRule, RuleAction

# Allow breaking single C-C bonds
cc_break = BondRule(
    name="single_cc_breakable",
    atom1_elem="C",
    atom2_elem="C",
    action=RuleAction.PREFER_BREAK,
    priority=5
)

# Protect C-N bonds
cn_keep = BondRule(
    name="protect_cn",
    atom1_elem="C",
    atom2_elem="N",
    action=RuleAction.PREFER_KEEP,
    priority=8
)
```

## Creating Custom Rule Classes

For more complex logic, extend the `FragmentationRule` base class:

```python
from autofragment.rules import FragmentationRule, RuleAction

class DistanceRule(FragmentationRule):
    """Break bonds longer than a threshold."""
    
    name = "distance_rule"
    
    def __init__(self, threshold=2.0, priority=None):
        super().__init__(priority=priority or self.PRIORITY_MEDIUM)
        self.threshold = threshold
    
    def applies_to(self, bond, system):
        # bond is a tuple of (atom1_idx, atom2_idx)
        dist = system.get_distance(bond[0], bond[1])
        return dist > self.threshold
    
    def action(self):
        return RuleAction.PREFER_BREAK
```

## Advanced: SMARTS Patterns

While SMARTS matching is not built into the core (to avoid dependency overhead), you can easily wrap RDKit or other libraries:

```python
try:
    from rdkit import Chem
except ImportError:
    pass

class SMARTSRule(FragmentationRule):
    def __init__(self, pattern_smarts, **kwargs):
        super().__init__(**kwargs)
        self.pattern = Chem.MolFromSmarts(pattern_smarts)
    
    def applies_to(self, bond, system):
        # Implement mapping between autofragment and RDKit
        return False
```

## Building a RuleSet

Combine multiple rules into a `RuleSet` for organization:

```python
from autofragment.rules import RuleSet, AromaticRingRule

my_rules = RuleSet(name="custom_polymer")
my_rules.add(AromaticRingRule())
my_rules.add(cc_break)
my_rules.add(DistanceRule(threshold=1.8))

# Or create from list
my_rules = RuleSet.from_rules([
    cc_break,
    cn_keep,
    DistanceRule()
])
```

## Using Custom Rules

Pass the rules to a `RuleEngine` to evaluate them:

```python
from autofragment.rules import RuleEngine

engine = RuleEngine(my_rules.rules)

# Check a specific bond
action = engine.evaluate_bond((0, 1), system)
```

## Priority System

Higher priority rules are evaluated first. If multiple rules apply to the same bond, **the most restrictive action wins**.

| Constant | Value | Typical Use |
|----------|-------|-------------|
| `PRIORITY_CRITICAL` | 1000 | Aromatic rings, multiple bonds |
| `PRIORITY_HIGH` | 800 | Metal coordination |
| `PRIORITY_MEDIUM` | 500 | Default user rules |
| `PRIORITY_LOW` | 200 | Weak preferences |

```python
# Override default priority
rule = DistanceRule(priority=999)
```

## Example: Polymer Rule Set

```python
from autofragment.rules import (
    RuleSet, BondRule, RuleAction, AromaticRingRule
)

polymer_rules = RuleSet(name="polymer")

# Protect aromatic rings (built-in)
polymer_rules.add(AromaticRingRule())

# Prefer breaking backbone C-C
polymer_rules.add(BondRule(
    name="backbone_cc",
    atom1_elem="C",
    atom2_elem="C",
    action=RuleAction.PREFER_BREAK,
    priority=500
))
```


## Debugging Rules

Check which rules affect specific bonds:

```python
engine = RuleEngine(my_rules.rules)

for bond in system.bonds:
    action = engine.evaluate_bond(bond, system)
    triggered = engine.get_triggered_rules(bond, system)
    print(f"Bond {bond}: {action}, rules: {[r.name for r in triggered]}")
```

## Best Practices

1. **Start with critical rules**: Always include aromatic and double bond protection
2. **Use appropriate priorities**: Critical > High > Medium > Low
3. **Test on small molecules**: Verify behavior before applying to large systems
4. **Log rule decisions**: Debug unexpected fragmentation
5. **Combine built-in and custom**: Extend existing rules rather than replacing
