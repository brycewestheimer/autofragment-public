# Fragmentation Rules

The Rules Engine is the core component that controls which bonds can or cannot be broken during molecular fragmentation. Rules encode chemical knowledge to ensure fragments remain chemically meaningful.

## Overview

The rules system provides:

- **Flexible rule definitions** via the `FragmentationRule` abstract base class
- **Priority-based evaluation** where critical rules are evaluated first
- **Conflict resolution** where the most restrictive action wins
- **Pre-built rules** for common chemistry, biology, and materials science

## Quick Start

```python
from autofragment.rules import (
    RuleEngine,
    RuleAction,
    AromaticRingRule,
    DoubleBondRule,
    PeptideBondRule,
)

# Create an engine with rules
engine = RuleEngine([
    AromaticRingRule(),      # Never break aromatic rings
    DoubleBondRule(),        # Never break double/triple bonds
    PeptideBondRule(),       # Prefer keeping peptide bonds
])

# Evaluate a specific bond
action = engine.evaluate_bond((0, 1), chemical_system)
if action == RuleAction.MUST_NOT_BREAK:
    print("This bond cannot be broken")

# Get all breakable bonds in a system
breakable = engine.get_breakable_bonds(chemical_system)
```

## Rule Actions

Rules return one of four actions, ordered from most to least restrictive:

| Action | Description | Use Case |
|--------|-------------|----------|
| `MUST_NOT_BREAK` | Bond must never be broken | Aromatic rings, double bonds |
| `PREFER_KEEP` | Prefer keeping, can break if necessary | Peptide bonds, metal coordination |
| `ALLOW` | No preference (neutral) | Default when no rules apply |
| `PREFER_BREAK` | Good fragmentation point | Alpha-beta carbon bonds |

When multiple rules apply to the same bond, **the most restrictive action wins**.

## Priority System

Rules have integer priorities (higher = evaluated first):

| Constant | Value | Typical Use |
|----------|-------|-------------|
| `PRIORITY_CRITICAL` | 1000 | Aromatic rings, proline rings |
| `PRIORITY_HIGH` | 800 | Metal coordination, peptide bonds |
| `PRIORITY_MEDIUM` | 500 | General rules (default) |
| `PRIORITY_LOW` | 200 | Hydrogen bonds |

```python
# Override default priority
rule = AromaticRingRule(priority=999)
print(rule.priority)  # 999
```

## Built-in Rules

### Common Chemical Rules

These rules apply universally to all molecular systems.

#### AromaticRingRule

Never break bonds within aromatic ring systems (benzene, naphthalene, pyridine, etc.).

```python
rule = AromaticRingRule()
# Priority: CRITICAL
# Action: MUST_NOT_BREAK
```

#### DoubleBondRule

Never break double or triple bonds. Configurable minimum bond order.

```python
rule = DoubleBondRule(min_order=1.5)  # Catches aromatic bonds too
# Priority: CRITICAL
# Action: MUST_NOT_BREAK
```

#### MetalCoordinationRule

Preserve metal-ligand coordination bonds. Supports custom metal sets.

```python
rule = MetalCoordinationRule(
    metals={"Fe", "Cu", "Zn"},
    rule_action=RuleAction.MUST_NOT_BREAK
)
# Priority: HIGH
```

#### FunctionalGroupRule

Keep common functional groups intact (carboxyl, nitro, phosphate, etc.).

```python
rule = FunctionalGroupRule()
# Default groups: carboxyl, nitro, sulfonate, phosphate, amino
```

### Biological Rules

Designed for proteins and nucleic acids. **These rules are configurable** to support different fragmentation strategies.

#### PeptideBondRule

Handle peptide bonds between amino acid residues.

```python
# Traditional FMO: keep peptide bonds
rule = PeptideBondRule(rule_action=RuleAction.PREFER_KEEP)

# Residue-based fragmentation: break at peptide bonds
rule = PeptideBondRule(rule_action=RuleAction.PREFER_BREAK)
```

#### DisulfideBondRule

Handle disulfide bridges (Cys-Cys S-S bonds).

```python
# Preserve structure
rule = DisulfideBondRule(rule_action=RuleAction.MUST_NOT_BREAK)

# Allow breaking for domain separation
rule = DisulfideBondRule(rule_action=RuleAction.ALLOW)
```

#### AlphaBetaCarbonRule

Mark C-alpha to C-beta bonds as preferred break points (sidechain separation).

```python
rule = AlphaBetaCarbonRule()
# Default action: PREFER_BREAK
```

#### ProlineRingRule

Never break the proline pyrrolidine ring (5-membered ring with N).

```python
rule = ProlineRingRule()
# Always: MUST_NOT_BREAK
```

#### HydrogenBondRule

Handle hydrogen bonds. **Defaults to ALLOW** because H-bonds often need to be broken.

```python
# For water clusters: allow breaking (default)
rule = HydrogenBondRule(rule_action=RuleAction.ALLOW)

# For secondary structure: prefer keeping
rule = HydrogenBondRule(rule_action=RuleAction.PREFER_KEEP)
```

### Materials Science Rules

For solid-state and extended materials systems.

#### SiloxaneBridgeRule

Handle Si-O-Si siloxane bridges in silica and zeolites.

```python
rule = SiloxaneBridgeRule(rule_action=RuleAction.PREFER_BREAK)
```

#### MOFLinkerRule

Preserve organic linker molecules in Metal-Organic Frameworks.

```python
rule = MOFLinkerRule()
# Protects C-C, C-N, C-O bonds within linkers
# Does NOT protect metal-linker bonds
```

#### MetalNodeRule

Preserve metal node clusters (Zn4O, Cu paddlewheels, etc.).

```python
rule = MetalNodeRule(cluster_distance=4.0)  # Angstroms
# Groups metals within 4Å as a cluster
```

#### PerovskiteOctahedralRule

Preserve MO₆ octahedra in perovskite structures.

```python
rule = PerovskiteOctahedralRule(
    b_site_metals={"Ti", "Zr", "Nb"},
    allow_corner_breaking=True
)
```

#### ZeoliteAcidSiteRule

Protect Brønsted acid sites (Al-O(H)-Si) in zeolites.

```python
rule = ZeoliteAcidSiteRule()
# Action: MUST_NOT_BREAK
```

#### SilanolRule

Preserve surface silanol groups (Si-OH).

```python
rule = SilanolRule()
# Action: MUST_NOT_BREAK
```

#### MetalCarboxylateRule

Identify and allow breaking of metal-carboxylate (M-O) bonds to separate nodes from organic linkers.

```python
rule = MetalCarboxylateRule(rule_action=RuleAction.PREFER_BREAK)
```

#### PolymerBackboneRule

Heuristic for identifying polymer backbone bonds.

```python
rule = PolymerBackboneRule()
```

## Creating Custom Rules

Extend `FragmentationRule` to create custom rules:

```python
from autofragment.rules import FragmentationRule, RuleAction

class MyCustomRule(FragmentationRule):
    """Custom rule for my specific system."""
    
    name = "my_custom_rule"  # Required: unique identifier
    
    def __init__(self, priority=None):
        super().__init__(priority=priority or self.PRIORITY_MEDIUM)
    
    def applies_to(self, bond, system):
        """Return True if this rule applies to the bond."""
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])
        
        # Your logic here
        return atom1["element"] == "X" and atom2["element"] == "Y"
    
    def action(self):
        """Return the action when this rule applies."""
        return RuleAction.MUST_NOT_BREAK
```

## RuleEngine API

The `RuleEngine` manages rules and evaluates bonds:

```python
engine = RuleEngine()

# Add/remove rules
engine.add_rule(AromaticRingRule())
engine.remove_rule("aromatic_ring")  # By name
engine.clear_rules()

# Query rules
rules = engine.get_rules()  # Sorted by priority
print(len(engine))  # Number of rules

# Evaluate bonds
action = engine.evaluate_bond((0, 1), system)
all_actions = engine.get_bond_actions(system)

# Find fragmentable bonds
breakable = engine.get_breakable_bonds(system)  # Not MUST_NOT_BREAK
preferred = engine.get_preferred_breaks(system)  # PREFER_BREAK only
```

## Best Practices

1. **Start with critical rules**: Always include `AromaticRingRule` and `DoubleBondRule`
2. **Be explicit about biological rules**: Choose appropriate actions for your fragmentation strategy
3. **Layer rules by priority**: Use CRITICAL for absolute constraints, HIGH for strong preferences
4. **Test with simple systems**: Verify rule behavior on small molecules before applying to large systems
5. **Log rule decisions**: Check which rules are triggering for debugging

## Example: Protein Fragmentation

```python
from autofragment.rules import (
    RuleEngine,
    AromaticRingRule,
    DoubleBondRule,
    PeptideBondRule,
    DisulfideBondRule,
    ProlineRingRule,
    AlphaBetaCarbonRule,
)

# FMO-style: fragment at alpha carbons, keep peptide bonds
fmo_engine = RuleEngine([
    AromaticRingRule(),                           # Protect aromatic sidechains
    DoubleBondRule(),                             # Protect double bonds
    ProlineRingRule(),                            # Protect proline rings
    DisulfideBondRule(),                          # Keep disulfides
    PeptideBondRule(rule_action=RuleAction.PREFER_KEEP),
    AlphaBetaCarbonRule(),                        # Break at C-alpha
])

# Get fragmentation points
breakable = fmo_engine.get_breakable_bonds(protein_system)
preferred = fmo_engine.get_preferred_breaks(protein_system)
```

## Example: MOF Fragmentation

```python
from autofragment.rules import (
    RuleEngine,
    AromaticRingRule,
    DoubleBondRule,
    MetalCoordinationRule,
    MOFLinkerRule,
    MetalNodeRule,
)

mof_engine = RuleEngine([
    AromaticRingRule(),        # Protect aromatic linkers
    DoubleBondRule(),          # Protect carboxylate C=O
    MOFLinkerRule(),           # Keep organic linkers intact
    MetalNodeRule(),           # Keep metal clusters intact
    # Metal-linker bonds will be the break points
])
```
