# Python API

## Quick API

```python
import autofragment as af

tree = af.partition_xyz("water64.xyz", n_fragments=4)
tree.to_json("partitioned.json")

for frag in tree.fragments:
    print(f"Fragment: {frag.id}, atoms: {len(frag.symbols)}")
```

## Partitioner objects

```python
import autofragment as af

partitioner = af.MolecularPartitioner(
    n_fragments=4,
    method="kmeans",  # or "kmeans_constrained" with pip install autofragment[balanced]
)

system = af.io.read_xyz("water64.xyz")
tree = partitioner.partition(system)

# If you need isolated molecules:
molecules = af.io.read_xyz_molecules("water64.xyz")
```

## Tiered (Hierarchical) Partitioning

For large systems, tiered partitioning groups molecules into primary fragments, then sub-partitions each into secondary (and optionally tertiary) fragments:

```python
import autofragment as af

# 2-tier: 4 primary fragments, each with 4 sub-fragments
partitioner = af.MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4, method="kmeans"
)
system = af.io.read_xyz("water64.xyz")
tree = partitioner.partition(system)

print(f"Primary fragments: {tree.n_primary}")
print(f"Total fragments: {tree.n_fragments}")  # 4 + 16 = 20
print(f"Hierarchical: {tree._is_hierarchical}")

for pf in tree.fragments:
    print(f"  {pf.id}: {len(pf.fragments)} sub-fragments")
    for sf in pf.fragments:
        print(f"    {sf.id}: {sf.n_atoms} atoms")
```

### 3-Tier Partitioning

```python
partitioner = af.MolecularPartitioner(
    tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
)
tree = partitioner.partition(system)
# Hierarchy: PF1 -> PF1_SF1 -> PF1_SF1_TF1
```

### Seeding Strategies

Control k-means initialization with seeding strategies:

```python
# Global seeding strategy
partitioner = af.MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4,
    method="kmeans", init_strategy="pca"
)

# Per-tier overrides
partitioner = af.MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4,
    method="kmeans",
    init_strategy="pca",             # default for all tiers
    init_strategy_secondary="radial" # override for secondary tier
)
```

Available strategies: `"halfplane"`, `"pca"`, `"axis"`, `"radial"`.

### Quick API (Tiered)

```python
tree = af.partition_xyz(
    "water64.xyz", tiers=2, n_primary=4, n_secondary=4
)
tree.to_json("tiered.json")
```

## Topology-Aware Selection

Use reusable neighborhood growth around seed atoms for QM/MM or broader partitioning workflows.

```python
from autofragment.partitioners import (
    QMMMPartitioner,
    TopologySelection,
)

system = af.io.read_pdb("protein.pdb")

qm_selection = TopologySelection(
    seed_atoms={10, 11, 12},
    mode="graph",        # or "euclidean"
    hops=2,
    layers=2,
    k_per_layer=8,
    expand_residues=True,
    bond_policy="infer", # or "strict"
)

partitioner = QMMMPartitioner(qm_selection=qm_selection, buffer_radius=5.0)
result = partitioner.partition(system)

print(len(result.qm_atoms), len(result.buffer_atoms), len(result.mm_atoms))
```

## Molecular Topology Refinement

`MolecularPartitioner` can optionally refine cluster assignments using topology overlap around representative molecules.

```python
from autofragment.partitioners import MolecularPartitioner

partitioner = MolecularPartitioner(
    n_fragments=8,
    method="kmeans_constrained",  # requires pip install autofragment[balanced]
    topology_refine=True,
    topology_mode="graph",
    topology_hops=1,
    topology_bond_policy="infer",
)

tree = partitioner.partition(af.io.read_xyz("water64.xyz"))
```

## Rules Engine

Use the rules engine to control which bonds can be broken:

```python
import autofragment as af
from autofragment.rules import (
    RuleEngine,
    RuleAction,
    AromaticRingRule,
    DoubleBondRule,
    PeptideBondRule,
)
from autofragment.core.types import ChemicalSystem

# Isolated molecules for rules evaluation
molecules = af.io.read_xyz_molecules("water64.xyz")

# Create an engine with common rules
engine = RuleEngine([
    AromaticRingRule(),      # Never break aromatic rings (CRITICAL priority)
    DoubleBondRule(),        # Never break double/triple bonds (CRITICAL)
    PeptideBondRule(),       # Prefer keeping peptide bonds (HIGH)
])

# Create a chemical system from isolated molecules (explicit boundary)
system = ChemicalSystem.from_molecules(molecules)

# Evaluate a specific bond
action = engine.evaluate_bond((0, 1), system)
print(f"Bond (0,1) action: {action}")

# Get all breakable bonds (action != MUST_NOT_BREAK)
breakable = engine.get_breakable_bonds(system)
print(f"Breakable bonds: {len(breakable)}")

# Get preferred break points (action == PREFER_BREAK)
preferred = engine.get_preferred_breaks(system)
```

### Configurable Biological Rules

Biological rules accept custom actions:

```python
from autofragment.rules import PeptideBondRule, DisulfideBondRule, RuleAction

# Traditional FMO: keep peptide bonds, break at alpha carbon
engine = RuleEngine([
    PeptideBondRule(rule_action=RuleAction.PREFER_KEEP),
])

# Residue-based: break at peptide bonds
engine = RuleEngine([
    PeptideBondRule(rule_action=RuleAction.PREFER_BREAK),
])

# Allow breaking of disulfide bridges
engine = RuleEngine([
    DisulfideBondRule(rule_action=RuleAction.ALLOW),
])
```

See the [Rules Documentation](rules.md) for complete details.
