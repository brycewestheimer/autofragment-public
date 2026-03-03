# Protein Fragmentation Guide

This guide covers the workflow for fragmenting protein structures using AutoFragment's `BioPartitioner`.

## Overview

Protein fragmentation presents unique challenges:
- Peptide backbone must be handled carefully
- Residue charges depend on pH
- Disulfide bonds create cross-links
- Water molecules around the protein need clustering

## Prerequisites

```python
from autofragment import ChemicalSystem, BioPartitioner
from autofragment.io.writers import write_gamess_fmo
```

Install the bio extra for mmCIF support:

```bash
pip install autofragment[bio]
```

## Step 1: Partition from mmCIF

The `BioPartitioner` reads mmCIF files and partitions proteins into residue-based fragments with water clustering.

```python
partitioner = BioPartitioner(
    water_clusters=10,
    water_cluster_method="kmeans",
    random_state=42,
    ph=7.4,
)

tree = partitioner.partition_file("1ubq.cif")

print(f"Created {len(tree.fragments)} fragments")
for frag in tree.fragments:
    print(f"  {frag.label}: {frag.n_atoms} atoms")
```

### CLI Usage

```bash
autofragment bio --input protein.cif --output partitioned.json --water-clusters 10
```

## Step 2: Generate FMO Input

```python
from autofragment.io.writers import write_gamess_fmo

write_gamess_fmo(
    tree.fragments,
    "protein_fmo.inp",
    method="MP2",
    basis="6-31G*",
)
```

## Step 3: Charge Assignment

### pH-Dependent Charges

Residue charges depend on pH:

| Residue | pKa | Charge at pH 7.4 |
|---------|-----|------------------|
| Asp | 3.9 | -1 |
| Glu | 4.1 | -1 |
| His | 6.0 | 0 (may vary) |
| Lys | 10.5 | +1 |
| Arg | 12.5 | +1 |
| Cys | 8.3 | 0 (unless in S-S) |
| Tyr | 10.5 | 0 |
| N-terminus | 9.7 | +1 |
| C-terminus | 2.0 | -1 |

## Complete Example

```python
#!/usr/bin/env python
"""Complete protein fragmentation workflow."""

from autofragment import BioPartitioner
from autofragment.io.writers import write_gamess_fmo


def fragment_protein(cif_file, output_file):
    # Configure and partition
    partitioner = BioPartitioner(
        water_clusters=10,
        water_cluster_method="kmeans",
        random_state=42,
        ph=7.4,
    )

    tree = partitioner.partition_file(cif_file)

    # Report
    print(f"Created {len(tree.fragments)} fragments:")
    for i, frag in enumerate(tree.fragments):
        print(f"  {i}: {frag.label}, {frag.n_atoms} atoms")

    # Write FMO input
    write_gamess_fmo(
        tree.fragments,
        output_file,
        method="MP2",
        basis="6-31G*",
    )
    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    fragment_protein("1ubq.cif", "ubiquitin_fmo.inp")
```

## Troubleshooting

### Common Issues

**Incorrect charges**: Check pH setting and histidine protonation states

**Missing atoms**: Ensure hydrogens are added before fragmentation (use `--add-implicit-hydrogens` flag)

**FMO convergence**: Try larger basis for initial guess

## Next Steps

- [Output Formats](output_formats.md): Alternative QC program formats
- [Algorithms Guide](algorithms.md): Details on clustering methods
