# Migration Guide

This guide covers the canonical model alignment introduced in Phase 10.5.

## ChemicalSystem Is the Canonical System Container

System-level APIs now expect `ChemicalSystem`. If you previously passed lists of
isolated molecules, use explicit conversion helpers:

```python
from autofragment.core import molecules_to_system, system_to_molecules

system = molecules_to_system(molecules)
# ... system-level workflows ...
molecules = system_to_molecules(system)
```

## XYZ Reader Changes

`io.read_xyz()` now returns a `ChemicalSystem` with molecule boundaries stored in
metadata. To keep the old behavior, use `io.read_xyz_molecules()`.

```python
from autofragment import io

system = io.read_xyz("water64.xyz")
molecules = io.read_xyz_molecules("water64.xyz")
```

## Partitioner Inputs

Partitioners now accept `ChemicalSystem` objects as the primary input. Passing
molecule lists still works, but emits a deprecation warning.

```python
from autofragment import io, MolecularPartitioner

system = io.read_xyz("water64.xyz")
partitioner = MolecularPartitioner(n_fragments=4)
result = partitioner.partition(system)
```

## Writing XYZ Files

`io.write_xyz()` accepts either a `ChemicalSystem` or a list of molecules. When
writing a system without molecule metadata, pass `atoms_per_molecule` to specify
chunking.

```python
from autofragment import io

io.write_xyz("output.xyz", system, atoms_per_molecule=3)
```
