# Quickstart Guide

Get started with AutoFragment in 5 minutes! This guide shows you how to install the library and run your first molecular fragmentation.

## Installation

Install AutoFragment using pip:

```bash
pip install autofragment
```

For development or additional features:

```bash
pip install autofragment[all]
```

### Optional Dependencies

Some features require additional packages:

```bash
# For balanced (constrained) clustering
pip install autofragment[balanced]

# For graph-based features
pip install autofragment[graph]

# For biological systems (mmCIF parsing via gemmi)
pip install autofragment[bio]

# For materials science (periodic systems)
pip install autofragment[matsci]
```

## Your First Fragmentation

### Load a System

```python
from autofragment import io

# From XYZ file
system = io.read_xyz("molecule.xyz")

# From PDB file (via FormatRegistry)
from autofragment.io.writers import FormatRegistry
system = FormatRegistry.read("protein.pdb")

# Print basic info
print(f"Loaded {system.n_atoms} atoms")
print(f"Bonds: {system.n_bonds}")
```

### Fragment the System

```python
from autofragment import MolecularPartitioner

# Create a partitioner with target number of fragments
partitioner = MolecularPartitioner(n_fragments=4)

# Fragment the system
tree = partitioner.partition(system)

# Inspect results
print(f"Created {len(tree.fragments)} fragments")
for i, frag in enumerate(tree.fragments):
    print(f"  Fragment {i}: {frag.n_atoms} atoms")
```

### Examine Fragment Details

```python
for fragment in tree.fragments:
    print(f"Fragment: {fragment.n_atoms} atoms")
    print(f"  Charge: {fragment.molecular_charge}")
    print(f"  Multiplicity: {fragment.molecular_multiplicity}")
    print(f"  Elements: {set(fragment.symbols)}")
```

## Export to a Quantum Chemistry Program

### GAMESS (FMO)

```python
from autofragment.io import write_gamess_fmo

write_gamess_fmo(
    tree.fragments,
    "output.inp",
    method="MP2",
    basis="6-31G*",
)
```

### Psi4

```python
from autofragment.io import write_psi4_sapt

write_psi4_sapt(
    tree.fragments,
    "output.dat",
    method="b3lyp",
    basis="cc-pVDZ",
)
```

### XYZ (Generic)

```python
from autofragment.io import write_xyz_fragments

write_xyz_fragments(tree.fragments, "fragments.xyz")
```

## Command-Line Interface

AutoFragment also provides a CLI:

```bash
# Basic fragmentation
autofragment single --input molecule.xyz --output partitioned.json --n-fragments 4

# Batch mode with consistent labeling
autofragment batch --reference ref.xyz --input-dir ./trajectory/ --output-dir ./partitioned/ --n-fragments 4

# Biological systems (mmCIF)
autofragment bio --input protein.cif --output partitioned.json

# See all options
autofragment --help
```

## Common Workflows

### Water Cluster (MBE)

```python
from autofragment import io, MolecularPartitioner

# Load water cluster
cluster = io.read_xyz("water20.xyz")

# Fragment into groups of waters
partitioner = MolecularPartitioner(n_fragments=4, method="kmeans")
tree = partitioner.partition(cluster)

print(f"Fragments: {len(tree.fragments)}")
```

### Tiered Hierarchical Fragmentation

For large systems, use tiered partitioning to create a hierarchy of fragments:

```python
from autofragment import MolecularPartitioner

partitioner = MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4, method="kmeans"
)
tree = partitioner.partition(cluster)

print(f"Primary fragments: {tree.n_primary}")
for pf in tree.fragments:
    print(f"  {pf.id}: {len(pf.fragments)} sub-fragments, {pf.n_atoms} atoms")
```

### Protein Fragmentation

```python
from autofragment import BioPartitioner

# Partition from mmCIF directly
partitioner = BioPartitioner()
tree = partitioner.partition_file("1ubq.cif")
```

## Configuration

### Clustering Methods

AutoFragment supports several clustering methods:

```python
from autofragment import MolecularPartitioner

# Standard k-means (default, no extra deps)
partitioner = MolecularPartitioner(n_fragments=4, method="kmeans")

# Constrained k-means for balanced fragments (requires autofragment[balanced])
partitioner = MolecularPartitioner(n_fragments=4, method="kmeans_constrained")

# Agglomerative clustering
partitioner = MolecularPartitioner(n_fragments=4, method="agglomerative")

# Spectral clustering
partitioner = MolecularPartitioner(n_fragments=4, method="spectral")
```

## Next Steps

Now that you've run your first fragmentation, explore more:

- **[Water Clusters Tutorial](water_clusters.md)**: Complete MBE workflow for water clusters
- **[Protein Fragmentation](proteins.md)**: Fragment proteins with biological rules
- **[Materials Science Guide](materials.md)**: Work with periodic systems
- **[Output Formats](output_formats.md)**: Export to different QC programs

## Getting Help

- **Documentation**: [Documentation Home](../index.md)
- **Development Guide**: [Development](../development.md)

## Troubleshooting

### Common Issues

**Import Error**: Make sure autofragment is installed in your environment:
```bash
pip show autofragment
```

**Missing Optional Features**: Install optional dependencies:
```bash
pip install autofragment[all]
```

**File Not Found**: Use absolute paths or check working directory:
```python
import os
print(os.getcwd())
```
