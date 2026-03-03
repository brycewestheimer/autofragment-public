# Core Concepts

AutoFragment is built upon a robust graph-based representation of molecular systems and a set of chemistry-aware utilities. This foundation ensures that fragmentation algorithms operate on physically meaningful structures.

## Molecular Graph

At the heart of the library is the `MolecularGraph` class, which wraps a NetworkX graph to represent atoms as nodes and bonds as edges.

```python
from autofragment.core.graph import MolecularGraph
from autofragment.core.types import Atom

# Methane example
atoms = [
    Atom("C", [0.0, 0.0, 0.0]),
    Atom("H", [0.6, 0.6, 0.6]),
    # ... more hydrogens
]

# Create graph and infer bonds automatically
mg = MolecularGraph.from_molecules([atoms], infer_bonds=True)

print(f"Atoms: {mg.n_atoms}, Bonds: {mg.n_bonds}")
```

### Features

-   **Bond Inference**: Automatically detects bonds based on inter-atomic distances and covalent radii.
-   **Ring Detection**: Identifies rings (cycles) crucial for aromaticity and fragmentation rules.
-   **Bridge Detection**: Finds "bridges" or cut-edges, which are often good candidates for fragmentation points.
-   **Subgraph Extraction**: Efficiently extract specific fragments or regions as independent graphs.

```python
# Check for rings
rings = mg.find_rings()

# Check if a bond is a bridge (removable without disconnecting the graph component)
is_bridge = mg.is_bridge(atom1_idx, atom2_idx)
```

## Chemistry Utilities

AutoFragment includes a built-in chemistry engine (`autofragment.core.chemistry`) to handle fundamental chemical properties.

### Key Capabilities

-   **Periodic Table Data**: Masses, valence electrons, electronegativity, and covalent radii.
-   **Bond Order Inference**: Infers Single, Double, Triple, and Aromatic (1.5) bond orders from geometry.
-   **Aromaticity Detection**: Uses Huckel's rule heuristics and bond order analysis to detect aromatic systems (e.g., Benzene).
-   **Formal Charge**: Estimates formal charges on atoms based on connectivity and valence.

```python
from autofragment.core import chemistry

order = chemistry.infer_bond_order("C", "C", distance=1.40)
# Returns 1.5 for aromatic C-C bond
```

## Data Structures

### ChemicalSystem

`ChemicalSystem` is the canonical representation of a full system: all atoms,
optional bonds, metadata, and lattice information. Public APIs accept a
`ChemicalSystem` for system-level operations.

### Molecule

`Molecule` is a lightweight helper for isolated fragments (e.g., single waters)
and geometry utilities. Conversions between `ChemicalSystem` and molecule lists
are explicit via `system_to_molecules` and `molecules_to_system`.

### Fragment

A `Fragment` represents a subset of a molecular system. It carries:
-   **Symbols & Geometry**: The atomic make-up.
-   **Child Fragments**: Optional `fragments` field for hierarchical nesting. A leaf fragment (`is_leaf == True`) holds atoms directly; a non-leaf fragment contains child `Fragment` objects.
-   **Metadata**: `molecular_charge`, `molecular_multiplicity`, and methods for `layer` (QM/MM) assignment.
-   **Graph Awareness**: Can map back to the original `MolecularGraph`.

The `n_atoms` property recurses through children, so it works correctly at any level of the hierarchy.

### FragmentTree

The `FragmentTree` is the primary output container. It holds:
-   **Fragments**: The list of resulting fragments (flat or hierarchical).
-   **Interfragment Bonds**: Explicit records of bonds that were cut, allowing for detailed analysis or restoration (e.g., capping).
-   **Provenance**: Metadata about the source file and algorithm used.

For hierarchical trees (produced by tiered partitioning), `FragmentTree` provides:
-   `n_primary`: Number of top-level fragments.
-   `_is_hierarchical`: Whether any fragment has children.
-   `n_fragments`: Total count across all hierarchy levels.

### Fragmentation Scheme

`FragmentationScheme` holds the algorithm configuration used to generate a
`FragmentTree`.

## Fragmentation Rules

The `autofragment.rules` module provides a **Rules Engine** that determines which bonds can be broken during fragmentation. Rules encode chemical knowledge to ensure fragments remain chemically meaningful.

### Rule Actions

Rules return one of four actions (most to least restrictive):

1. **MUST_NOT_BREAK**: Bond must never be broken (aromatic rings, double bonds)
2. **PREFER_KEEP**: Prefer keeping, but can break if necessary (peptide bonds)
3. **ALLOW**: No preference (default)
4. **PREFER_BREAK**: Good fragmentation point (alpha-beta carbon bonds)

### Built-in Rules

- **Common**: `AromaticRingRule`, `DoubleBondRule`, `MetalCoordinationRule`, `FunctionalGroupRule`
- **Biological** (configurable): `PeptideBondRule`, `DisulfideBondRule`, `ProlineRingRule`, `HydrogenBondRule`
- **Materials Science**: `SiloxaneBridgeRule`, `MOFLinkerRule`, `MetalNodeRule`, `PerovskiteOctahedralRule`

See the [Rules Documentation](rules.md) for complete details on creating and using fragmentation rules.
