# Biological System Support

AutoFragment (v0.5+) includes specialized support for partitioning biological macromolecules like proteins and nucleic acids.

## Amino Acids & Proteins

Documentation of the 20 standard amino acids, plus common variants, is built-in.

### Charge Handling

AutoFragment includes a sophisticated pH-dependent charge calculation engine.

*   **Standard Charges:** Default charges for all 20 amino acids.
*   **pH Dependency:** Uses the Henderson-Hasselbalch equation to calculate partial charges at arbitrary pH values.
    *   Example: Histidine at pH 7.4 has a partial positive charge (~+0.1).
*   **Terminals:** N- and C- termini pKa values are average-adjusted based on the residue type (e.g., Proline N-term).
*   **PTMs:** Support for Post-Translational Modifications:
    *   Phosphorylation (SEP, TPO, PTR)
    *   Acetylation (ALY)
    *   Methylation (MLY, M3L)

### BioPartitioner

The `BioPartitioner` class is enhanced to use this data:

```python
from autofragment import BioPartitioner

# Partition with physiological charges
partitioner = BioPartitioner(pdb_file="protein.pdb", ph=7.4)
fragments = partitioner.partition()
```

### Secondary Structure

Partitioning respects secondary structure elements if provided (HELIX/SHEET records in PDB/CIF). It avoids breaking consecutive residues within a helix or beta-sheet strand.

## Nucleic Acids (DNA/RNA)

Specialized support for DNA and RNA fragmentation is provided via `NucleicPartitioner`.

### Features

*   **Nucleotide Data:** Comprehensive definitions for DNA (dA, dT, dG, dC) and RNA (A, U, G, C) bases.
*   **Backbone Charges:** Phosphodiester backbone is assigned -1 charge per nucleotide.
*   **Base Pairs:** Can detect Watson-Crick (A-T, G-C, A-U) and Wobble (G-U) base pairs to keep paired bases in the same partition if desired.

### Usage

```python
from autofragment import NucleicPartitioner

# Partition DNA, keeping base pairs together
partitioner = NucleicPartitioner(
    nucleotides_per_fragment=3,
    preserve_base_pairs=True
)
```

### Partitioning Strategies

1.  **Backbone (Default):** Breaks the phosphate backbone, keeping each base attached to its sugar and phosphate.
2.  **Base Separation:** separates nucleobases from the sugar-phosphate backbone (useful for specific QM studies).
3.  **Hybrid:** Flexible fragmentation based on size.
