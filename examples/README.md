# AutoFragment Examples

This directory contains examples and test data for `autofragment`.

## Notebooks

Interactive Jupyter notebooks demonstrating core features:

1. **[Quickstart](notebooks/01_quickstart.ipynb)**: A 5-minute introduction to fragmentation concepts.
2. **[Water Cluster MBE](notebooks/02_water_mbe.ipynb)**: Many-Body Expansion (MBE) fragmentation of water clusters.
3. **[Protein EFMO](notebooks/03_protein_efmo.ipynb)**: Fragmenting proteins for GAMESS EFMO calculations.
4. **[Algorithm Comparison](notebooks/04_compare_algorithms.ipynb)**: Benchmarking different partitioning algorithms.
5. **[Custom Scoring](notebooks/05_custom_scoring.ipynb)**: Implementing custom partitioning logic.
6. **[Zeolite Fragmentation](notebooks/06_zeolite_fragment.ipynb)**: Dealing with periodic systems and CIF files.
7. **[QM/MM Setup](notebooks/07_qmmm_setup.ipynb)**: Defining QM and MM regions.

## Data

Test data files used in the examples:

- `1l2y.pdb`: Trp-cage protein structure (20 residues).
- `1l2y.cif`: Trp-cage protein structure (mmCIF format).
- `sodalite.cif`: Sodalite zeolite unit cell.
- `aspirin.xyz`: Aspirin molecule.
- `caffeine.xyz`: Caffeine molecule.
- `dna_hairpin.pdb`: DNA hairpin structure.
- `water_64.xyz`: Cluster of 64 water molecules.

## Usage

To run the notebooks, ensure you have `jupyter` installed:

```bash
pip install jupyter
```

Then start the notebook server:

```bash
jupyter notebook notebooks/
```
