# Multi-Level Methods & QM/MM Partitioning

AutoFragment provides comprehensive support for multi-level computational methods, including ONIOM-style calculations and QM/MM partitioning. This module enables sophisticated treatment of large molecular systems by dividing them into regions treated with different levels of theory.

## Overview

Multi-level methods allow you to:

- **Define computational layers** with different methods and basis sets
- **Partition systems into QM/MM regions** with automatic boundary detection
- **Position link atoms** at covalent boundaries using g-factor approaches
- **Generate point charges** for electrostatic embedding
- **Create ONIOM input** for Gaussian and GAMESS

## Quick Start

### Basic QM/MM Partitioning

```python
from autofragment.partitioners import (
    QMMMPartitioner,
    AtomSelection,
    DistanceSelection,
)

# Define QM region by explicit atom indices
selection = AtomSelection(atom_indices={0, 1, 2, 3, 4})

# Create partitioner with buffer region
partitioner = QMMMPartitioner(
    qm_selection=selection,
    buffer_radius=5.0,  # Angstrom
    link_scheme="hydrogen"
)

# Partition molecules
result = partitioner.partition(molecules)

# Access regions
print(f"QM atoms: {result.qm_atoms}")
print(f"Buffer atoms: {result.buffer_atoms}")
print(f"MM atoms: {result.mm_atoms}")
print(f"Link atoms: {len(result.link_atoms)}")
```

### ONIOM Calculations

```python
from autofragment.multilevel import ONIOMScheme

# Parse from Gaussian-style string
scheme = ONIOMScheme.from_string("ONIOM(B3LYP/6-31G*:UFF)")

# Set atoms for each layer
scheme.set_layer_atoms("high", {0, 1, 2, 3, 4})
scheme.set_layer_atoms("low", {5, 6, 7, 8, 9, 10})

# Generate input files
gaussian_input = scheme.to_gaussian_input()
# Output: ONIOM(B3LYP/6-31G*:UFF)

gamess_input = scheme.to_gamess_input()
```

## Computational Layers

### Layer Types

AutoFragment supports four layer types:

| Layer Type | Description | Use Case |
|------------|-------------|----------|
| `HIGH` | Highest level QM | Active site, reaction center |
| `MEDIUM` | Intermediate QM | First solvation shell |
| `LOW` | Low-level QM/SE | Extended environment |
| `MM` | Molecular mechanics | Bulk solvent |

### Creating Layers

```python
from autofragment.multilevel import (
    ComputationalLayer,
    LayerType,
    MultiLevelScheme,
)

# Create a high-level QM layer
qm_layer = ComputationalLayer(
    name="active_site",
    layer_type=LayerType.HIGH,
    method="B3LYP",
    basis_set="6-311G(2d,2p)",
    atom_indices={0, 1, 2, 3, 4},
    charge=-1,
    multiplicity=1,
)

# Create an MM layer
mm_layer = ComputationalLayer(
    name="environment",
    layer_type=LayerType.MM,
    method="AMBER",
    atom_indices={5, 6, 7, 8, 9, 10},
)

# Combine into a scheme
scheme = MultiLevelScheme(
    name="enzyme_qmmm",
    scheme_type="qmmm",
)
scheme.add_layer(qm_layer)
scheme.add_layer(mm_layer)

# Access layers
high = scheme.get_high_layer()
print(f"QM method: {high.method}/{high.basis_set}")
```

### Embedding Types

Control how layers interact:

```python
from autofragment.multilevel import EmbeddingType

# Electrostatic embedding (QM sees MM charges)
scheme = MultiLevelScheme(
    name="qmmm",
    embedding_type=EmbeddingType.ELECTROSTATIC,
)

# Mechanical embedding (no electronic coupling)
scheme.embedding_type = EmbeddingType.MECHANICAL

# Polarizable embedding (mutual polarization)
scheme.embedding_type = EmbeddingType.POLARIZABLE
```

## QM Region Selection

### Selection Strategies

AutoFragment provides multiple ways to define the QM region:

#### Atom Selection (Explicit)

```python
from autofragment.partitioners import AtomSelection

# Specify exact atom indices
selection = AtomSelection(atom_indices={0, 1, 2, 3, 4, 5})
```

#### Distance Selection (Spherical)

```python
from autofragment.partitioners import DistanceSelection
import numpy as np

# Select all atoms within 5 Å of a point
center = np.array([10.0, 15.0, 20.0])
selection = DistanceSelection(center=center, radius=5.0)
```

#### Residue Selection (Biological)

```python
from autofragment.partitioners import ResidueSelection

# Select by residue name (for proteins)
selection = ResidueSelection(residue_names=["HIS", "CYS", "ASP"])

# Or by residue number
selection = ResidueSelection(residue_numbers=[64, 68, 102])
```

#### Combined Selection

```python
from autofragment.partitioners import CombinedSelection

# Union: atoms matching any criterion
union_selection = CombinedSelection(
    [atom_sel, distance_sel],
    mode="union"
)

# Intersection: atoms matching all criteria
intersection_selection = CombinedSelection(
    [residue_sel, distance_sel],
    mode="intersection"
)
```

#### Topology Selection (Graph / Euclidean Layers)

Use topology-aware neighborhood growth from one or more seed atoms.

```python
from autofragment.partitioners import TopologySelection

# Graph-hop expansion from active-site seed atoms
topology_sel = TopologySelection(
    seed_atoms={120, 121, 122},
    mode="graph",        # or "euclidean"
    hops=2,               # graph mode depth
    layers=2,             # euclidean mode only
    k_per_layer=12,       # euclidean mode only
    expand_residues=True,
    bond_policy="infer", # or "strict" (requires explicit bonds)
)

partitioner = QMMMPartitioner(qm_selection=topology_sel, buffer_radius=5.0)
result = partitioner.partition(
    system,
    residue_names=residue_names,
    residue_numbers=residue_numbers,
    chains=chains,
)
```

`TopologySelection` uses the shared `TopologyNeighborSelection` utility, which is also available for non-QM/MM workflows.

### Reusable Topology Utility

```python
from autofragment.partitioners import TopologyNeighborSelection

selector = TopologyNeighborSelection(
    seed_atoms={0, 1, 2},
    mode="graph",
    hops=1,
    bond_policy="infer",
)

selection_result = selector.select(coords, elements, bonds=bonds)
print(selection_result.selected_atoms)
print(selection_result.shells)
```

## Link Atoms

When covalent bonds cross the QM/MM boundary, link atoms (typically hydrogen) cap the QM region.

### Automatic Link Atom Placement

```python
from autofragment.multilevel import (
    calculate_g_factor,
    create_link_atoms_for_cut_bonds,
)

# G-factor determines link atom position
# R_link = R_QM + g * (R_MM - R_QM)
g = calculate_g_factor("C", "C", "H")  # ~0.708

# Create link atoms for cut bonds
link_atoms = create_link_atoms_for_cut_bonds(
    coords=coordinates,
    elements=elements,
    cut_bonds=[(0, 5), (3, 8)],  # (qm_idx, mm_idx)
    link_element="H"
)

for la in link_atoms:
    print(f"Link H at QM atom {la.qm_atom_index}, replacing MM atom {la.mm_atom_index}")
    print(f"Position: {la.position}")
```

### Bond Length Data

Common bond lengths used for g-factor calculations:

| Bond | Length (Å) |
|------|------------|
| C-H | 1.09 |
| C-C | 1.54 |
| C-N | 1.47 |
| N-H | 1.01 |
| C-O | 1.43 |

## Point Charge Embedding

For electrostatic QM/MM, MM atoms are represented as point charges:

```python
from autofragment.multilevel import PointChargeEmbedding

# Create embedding generator
embedding = PointChargeEmbedding(
    charge_scheme="default",
    exclude_link_atoms=True,  # Don't double-count
)

# Generate point charges
charges = embedding.generate_charges(
    coords=coordinates,
    elements=elements,
    mm_atoms=result.mm_atoms,
    link_mm_atoms={la.mm_atom_index for la in result.link_atoms},
)

# Get arrays for QM program
positions, charge_values = embedding.to_arrays(charges)

# Generate GAMESS input
gamess_efrag = embedding.to_gamess_format(charges)

# Generate Gaussian-style charges
gaussian_charges = embedding.to_gaussian_format(charges)
```

## Layer Assignment Algorithms

Automatically assign atoms to layers:

```python
from autofragment.multilevel import (
    assign_by_distance,
    assign_by_residue,
    assign_by_element,
)

# Distance-based: atoms within cutoffs from center
layers = assign_by_distance(
    coords=coordinates,
    elements=elements,
    center_indices={0, 1, 2},
    cutoffs=[5.0, 10.0],  # Creates 3 layers
)

# Residue-based: group by residue names
qm_atoms = assign_by_residue(
    coords=coordinates,
    residue_names=residue_names,
    qm_residues={"HIS", "CYS"},
)

# Element-based: group by element type
layers = assign_by_element(
    coords=coordinates,
    elements=elements,
    element_layers={"Fe": "high", "N": "medium"},
)
```

## ONIOM Input Generation

### Two-Layer ONIOM

```python
from autofragment.multilevel import create_oniom_scheme

scheme = create_oniom_scheme(
    high_method="B3LYP",
    high_basis="6-31G*",
    high_atoms={0, 1, 2, 3, 4},
    low_method="UFF",
    low_atoms={5, 6, 7, 8, 9, 10},
)

print(scheme.to_gaussian_input())
# ONIOM(B3LYP/6-31G*:UFF)
```

### Three-Layer ONIOM

```python
scheme = ONIOMScheme.from_string(
    "ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G*:AMBER)"
)

# Access layers
print(scheme.high_method)   # CCSD(T)
print(scheme.medium_method) # B3LYP
print(scheme.low_method)    # AMBER
```

## Converting Results

### To MultiLevelScheme

```python
# Convert QMMMResult to MultiLevelScheme
scheme = result.to_multilevel_scheme(
    qm_method="B3LYP",
    qm_basis="6-31G*",
    mm_method="AMBER",
    name="enzyme_active_site"
)
```

### To FragmentTree

```python
# For compatibility with other AutoFragment workflows
tree = partitioner.partition_to_fragment_tree(molecules)
tree.to_json("qmmm_fragments.json")
```

## Serialization

All data structures support JSON serialization:

```python
# Save
data = result.to_dict()
import json
with open("qmmm_result.json", "w") as f:
    json.dump(data, f)

# Load
from autofragment.partitioners import QMMMResult
with open("qmmm_result.json") as f:
    result = QMMMResult.from_dict(json.load(f))
```

## API Reference

### Classes

- {py:class}`autofragment.multilevel.ComputationalLayer`
- {py:class}`autofragment.multilevel.MultiLevelScheme`
- {py:class}`autofragment.multilevel.ONIOMScheme`
- {py:class}`autofragment.multilevel.LinkAtom`
- {py:class}`autofragment.multilevel.PointCharge`
- {py:class}`autofragment.multilevel.PointChargeEmbedding`
- {py:class}`autofragment.partitioners.QMMMPartitioner`
- {py:class}`autofragment.partitioners.QMMMResult`

### Selection Classes

- {py:class}`autofragment.partitioners.AtomSelection`
- {py:class}`autofragment.partitioners.DistanceSelection`
- {py:class}`autofragment.partitioners.ResidueSelection`
- {py:class}`autofragment.partitioners.CombinedSelection`
- {py:class}`autofragment.partitioners.TopologySelection`
- {py:class}`autofragment.partitioners.TopologyNeighborSelection`

### Functions

- {py:func}`autofragment.multilevel.assign_by_distance`
- {py:func}`autofragment.multilevel.assign_by_residue`
- {py:func}`autofragment.multilevel.assign_by_element`
- {py:func}`autofragment.multilevel.calculate_g_factor`
- {py:func}`autofragment.multilevel.create_link_atoms_for_cut_bonds`
