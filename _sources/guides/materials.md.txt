# Materials Science Guide

This guide covers the fragmentation of materials science systems, including surfaces and other periodic systems.

## Prerequisites

```python
from autofragment import ChemicalSystem
from autofragment.partitioners import SurfacePartitioner
from autofragment.io.readers import read_cif, read_poscar
```

## Loading Periodic Systems

### From CIF Files

```python
system = read_cif("structure.cif")
print(f"Atoms: {system.n_atoms}")
if system.lattice:
    print(f"Lattice: {system.lattice}")
```

### From VASP Files

```python
system = read_poscar("POSCAR")
```

### Lattice Operations

```python
lattice = system.lattice
print(f"a, b, c: {lattice.lengths}")
print(f"alpha, beta, gamma: {lattice.angles}")
print(f"Volume: {lattice.volume:.2f} A^3")
```

## Surface Systems

```python
partitioner = SurfacePartitioner(
    surface_normal=2,         # z-axis
    n_surface_layers=2,
    n_bulk_layers=3,
)
fragments = partitioner.partition(slab)
```

## Future Work

The following partitioners are planned for future releases but not yet implemented:

- **MOFPartitioner**: Metal-Organic Framework decomposition into nodes and linkers
- **ZeolitePartitioner**: Aluminosilicate framework partitioning

Similarly, the following writers are planned:

- **VASP writer**: POSCAR output for periodic fragments
- **Quantum ESPRESSO writer**: pw.in output for periodic DFT

## Tips

1. Use large enough supercells to avoid periodic image artifacts
2. Keep complete coordination spheres in fragments
3. Protect aromatic rings and functional groups
4. Use appropriate charges for accurate embedding
