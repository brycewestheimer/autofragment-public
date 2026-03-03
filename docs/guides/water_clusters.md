# Water Cluster Fragmentation Tutorial

This tutorial demonstrates a complete workflow for fragmenting water clusters and
writing quantum chemistry input files for fragment-based calculations. Water
clusters are ideal starting points because individual water molecules are natural
fragments with well-defined boundaries.

## Overview

We'll cover:
1. Loading a water cluster from an XYZ file
2. Partitioning into fragment groups with `MolecularPartitioner`
3. Choosing a clustering method (including balanced partitioning)
4. Tiered hierarchical fragmentation for large clusters
5. Writing fragment-based input files for GAMESS, Psi4, and other programs
6. Exporting results as QCSchema JSON

## Prerequisites

```python
import autofragment as af
from autofragment.partitioners import MolecularPartitioner
from autofragment.io.writers import (
    write_gamess_fmo,
    write_psi4_sapt,
    write_psi4_fragment,
)
```

## Step 1: Load Water Cluster

### From XYZ File

`read_xyz` auto-detects water molecules (groups of 3 atoms each) by default:

```python
system = af.io.read_xyz("water20.xyz")

print(f"Total atoms: {len(system.atoms)}")
print(f"Molecules: {len(system.metadata.get('molecule_atom_indices', []))}")
```

### Customizing Molecule Detection

For non-water systems or to disable water validation:

```python
# Explicit atoms-per-molecule (e.g., methanol with 6 atoms each)
system = af.io.read_xyz("methanol_cluster.xyz", atoms_per_molecule=6)

# Skip water geometry validation
system = af.io.read_xyz("water20.xyz", validate_water=False)
```

### Example XYZ Format

```text
60
(H2O)20 cluster
O    0.000    0.000    0.000
H    0.957    0.000    0.000
H   -0.240    0.927    0.000
O    2.800    0.000    0.000
...
```

## Step 2: Partition into Fragments

### Basic Partitioning

```python
partitioner = MolecularPartitioner(n_fragments=4, method="kmeans")
tree = partitioner.partition(system)

print(f"Fragments: {tree.n_fragments}")
print(f"Total atoms: {tree.n_atoms}")

for frag in tree.fragments:
    print(f"  {frag.id}: {len(frag.symbols)} atoms")
```

### Available Clustering Methods

`MolecularPartitioner` supports several clustering algorithms via the `method` parameter:

| Method | Description | Extra required |
|--------|-------------|----------------|
| `"kmeans"` | Standard k-means (default) | — |
| `"kmeans_constrained"` | K-means with balanced cluster sizes | `pip install autofragment[balanced]` |
| `"agglomerative"` | Hierarchical agglomerative clustering | — |
| `"spectral"` | Spectral clustering | `pip install autofragment[graph]` |
| `"gmm"` | Gaussian Mixture Model | — |
| `"birch"` | BIRCH clustering | — |
| `"geom_planes"` | Geometric partitioning by cutting planes | — |

### Balanced Partitioning with `kmeans_constrained`

For many-body expansion (MBE) and FMO workflows, equally-sized fragments are
often critical for load balancing and accuracy. The `kmeans_constrained` method
enforces balanced cluster sizes:

```bash
pip install "autofragment[balanced]"
```

```python
partitioner = MolecularPartitioner(
    n_fragments=4,
    method="kmeans_constrained",
)
tree = partitioner.partition(system)

# Verify balanced sizes
sizes = [len(frag.symbols) for frag in tree.fragments]
print(f"Fragment sizes: {sizes}")
# e.g., [15, 15, 15, 15] for 20 waters into 4 fragments
```

You can also enable balanced partitioning explicitly with any method that supports it:

```python
partitioner = MolecularPartitioner(
    n_fragments=4,
    method="kmeans",
    strict_balanced=True,  # automatically uses kmeans_constrained
)
```

### Topology-Aware Refinement

For better chemical locality, enable topology refinement which reassigns
molecules based on neighborhood overlap:

```python
partitioner = MolecularPartitioner(
    n_fragments=4,
    method="kmeans",
    topology_refine=True,
    topology_mode="graph",
    topology_hops=1,
)
tree = partitioner.partition(system)
```

### Tiered Hierarchical Partitioning

For large clusters, tiered partitioning creates a hierarchy of fragments. This is
useful for hierarchical many-body expansion (HMBE) workflows where you need
multiple levels of spatial grouping:

```python
# 2-tier: 4 primary fragments, each with 4 sub-fragments
partitioner = MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4, method="kmeans"
)
tree = partitioner.partition(system)

print(f"Primary fragments: {tree.n_primary}")
print(f"Total fragments: {tree.n_fragments}")

for pf in tree.fragments:
    print(f"  {pf.id}: {len(pf.fragments)} sub-fragments")
    for sf in pf.fragments:
        print(f"    {sf.id}: {sf.n_atoms} atoms")
```

For even finer control, use 3-tier partitioning:

```python
partitioner = MolecularPartitioner(
    tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
)
tree = partitioner.partition(system)
# PF1 -> PF1_SF1 -> PF1_SF1_TF1, PF1_SF1_TF2, ...
```

### K-Means Seeding Strategies

Control the initial centroids for k-means clustering:

```python
# PCA-based seeding
partitioner = MolecularPartitioner(
    n_fragments=4, method="kmeans", init_strategy="pca"
)

# Per-tier seeding overrides (tiered mode)
partitioner = MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4,
    method="kmeans",
    init_strategy="pca",              # default for all tiers
    init_strategy_secondary="radial"  # override for sub-fragments
)
```

Available strategies: `"halfplane"`, `"pca"`, `"axis"`, `"radial"`.

## Step 3: Write Input Files

### GAMESS FMO

```python
write_gamess_fmo(
    tree.fragments,
    "water_fmo.inp",
    basis="aug-cc-pVDZ",
    method="MP2",
    runtype="energy",
    fmo_level=2,
    nbody=2,
)
```

### GAMESS EFMO

```python
from autofragment.io.writers import write_gamess_efmo

write_gamess_efmo(
    tree.fragments,
    "water_efmo.inp",
    basis="aug-cc-pVDZ",
)
```

### Psi4 SAPT (Dimer Interaction Energies)

For symmetry-adapted perturbation theory between fragment pairs:

```python
# Select two fragments for a SAPT calculation
dimer = [tree.fragments[0], tree.fragments[1]]

write_psi4_sapt(
    dimer,
    "water_sapt.dat",
    method="sapt2+",
    basis="aug-cc-pVDZ",
)
```

### Psi4 General Fragment Calculation

```python
write_psi4_fragment(
    tree.fragments,
    "water_frags.dat",
    method="mp2",
    basis="aug-cc-pVDZ",
    runtype="energy",
)
```

### Other Programs

autofragment also provides writers for Q-Chem, NWChem, ORCA, Molpro, Turbomole,
and CFOUR. See [Output Formats](output_formats.md) for the full list.

## Step 4: Export and Save Results

### QCSchema JSON

```python
# Save the full FragmentTree
tree.to_json("water20_fragments.json")

# Reload later
tree = af.FragmentTree.from_json("water20_fragments.json")
```

### QCSchema Writer

```python
from autofragment.io.writers import write_qcschema

write_qcschema(system, "water20_qcschema.json")
```

### XYZ Fragment Output

```python
from autofragment.io.writers import write_xyz_fragments

write_xyz_fragments(tree.fragments, "water20_fragments.xyz")
```

## Step 5: Quick API

For simple workflows, the convenience function handles reading and partitioning
in a single call:

```python
import autofragment as af

tree = af.partition_xyz("water20.xyz", n_fragments=4, method="kmeans")
tree.to_json("water20_fragments.json")

for frag in tree.fragments:
    print(f"Fragment {frag.id}: {len(frag.symbols)} atoms")
```

## Complete Example

```python
#!/usr/bin/env python
"""Fragment a water cluster and write GAMESS FMO input."""

import autofragment as af
from autofragment.partitioners import MolecularPartitioner
from autofragment.io.writers import write_gamess_fmo

def main():
    # Load
    system = af.io.read_xyz("water20.xyz")
    print(f"Loaded {len(system.atoms)} atoms")

    # Partition with balanced clusters
    partitioner = MolecularPartitioner(
        n_fragments=4,
        method="kmeans_constrained",  # requires autofragment[balanced]
    )
    tree = partitioner.partition(system)
    print(f"Created {tree.n_fragments} fragments")

    for frag in tree.fragments:
        print(f"  {frag.id}: {len(frag.symbols)} atoms")

    # Write GAMESS FMO input
    write_gamess_fmo(
        tree.fragments,
        "water20_fmo.inp",
        basis="aug-cc-pVDZ",
        method="MP2",
        fmo_level=2,
        nbody=2,
    )
    print("Wrote water20_fmo.inp")

    # Also save as JSON for later use
    tree.to_json("water20_fragments.json")

if __name__ == "__main__":
    main()
```

## Tips and Best Practices

1. **Basis set**: Use augmented basis sets (aug-cc-pVDZ, aug-cc-pVTZ) for accurate hydrogen bonding.
2. **Balanced fragments**: Use `method="kmeans_constrained"` (or `strict_balanced=True`) when equal-sized fragments matter for your workflow.
3. **Fragment count**: Start with fewer fragments and increase; more fragments means more n-body terms.
4. **Topology refinement**: Enable `topology_refine=True` for better spatial locality in larger clusters.
5. **Parallelization**: Fragment-based QC calculations are embarrassingly parallel — each fragment or pair can run independently.

## Next Steps

- [Algorithms Guide](../guides/algorithms.md): Details on each clustering method
- [Protein Fragmentation](proteins.md): Apply fragmentation to biomolecules
- [Output Formats](output_formats.md): Complete list of QC program writers
