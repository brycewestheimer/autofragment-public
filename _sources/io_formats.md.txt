# I/O Formats Guide

AutoFragment provides comprehensive I/O support for reading molecular structure files and writing fragmented systems to various quantum chemistry program formats.

## Overview

The I/O module (`autofragment.io`) supports:

- **11 input formats** for reading molecular structures
- **12 output formats** for writing fragment-annotated input files
- **FormatRegistry** for extensible format handling

## Reading Molecular Structures

### Quick Start

```python
from autofragment import io

# Read from various formats (auto-detection by extension)
system = io.read_pdb("protein.pdb")
system = io.read_mol2("ligand.mol2")
system = io.read_sdf("compounds.sdf")
system = io.read_xyz("water64.xyz")

# All readers return ChemicalSystem objects
print(f"Atoms: {system.n_atoms}")
print(f"Bonds: {system.n_bonds}")
```

For isolated molecule lists, use the explicit XYZ helper:

```python
molecules = io.read_xyz_molecules("water64.xyz")
```

### Supported Input Formats

| Format | Function | Extensions | Description |
|--------|----------|------------|-------------|
| PDB | `read_pdb()` | `.pdb`, `.ent` | Protein Data Bank with CONECT records |
| MOL2 | `read_mol2()` | `.mol2` | Tripos MOL2 with atom types and charges |
| SDF/MOL | `read_sdf()` | `.sdf`, `.mol` | MDL V2000/V3000 structure files |
| QCSchema | `read_qcschema()` | `.json` | MolSSI standard JSON format |
| GAMESS | `read_gamess_input()` | `.inp` | GAMESS input file geometry |
| Psi4 | `read_psi4_input()` | `.dat` | Psi4 molecule blocks |
| Q-Chem | `read_qchem_input()` | `.in` | Q-Chem $molecule sections |
| ORCA | `read_orca_input()` | `.inp` | ORCA *xyz blocks |
| NWChem | `read_nwchem_input()` | `.nw` | NWChem geometry blocks |
| VASP | `read_poscar()` | `.poscar`, `.contcar` | VASP POSCAR/CONTCAR files |
| CIF | `read_cif()` | `.cif`, `.mmcif` | Crystallographic Information Files |

### PDB Reader

The PDB reader supports:
- ATOM and HETATM records
- CONECT records for explicit bonds
- Multi-MODEL files (select with `model=N`)
- Distance-based bond inference

```python
from autofragment.io import read_pdb

# Read with explicit bonds only
system = read_pdb("protein.pdb", infer_bonds=False)

# Read specific model from NMR ensemble
system = read_pdb("ensemble.pdb", model=5)

# Access metadata
print(system.metadata["source_file"])
```

### MOL2 Reader

The MOL2 reader handles:
- Tripos atom type mapping (C.3, N.ar, etc.)
- Bond orders (1=single, 2=double, ar=aromatic)
- Partial charges
- Multi-molecule files

```python
from autofragment.io import read_mol2

system = read_mol2("ligand.mol2")

# Charges are stored per atom
for atom in system.atoms:
    print(f"{atom.symbol}: charge={atom.charge}")
```

### SDF/MOL Reader

Supports both MDL V2000 (fixed-width) and V3000 (keyword-based) formats:

```python
from autofragment.io import read_sdf, read_sdf_multi

# Read first molecule
system = read_sdf("compound.sdf")

# Read all molecules from multi-molecule file
systems = read_sdf_multi("library.sdf")
for i, sys in enumerate(systems):
    print(f"Molecule {i}: {sys.n_atoms} atoms")
```

### QCSchema Reader

Reads MolSSI QCSchema JSON format with automatic Bohr→Angstrom conversion:

```python
from autofragment.io import read_qcschema

system = read_qcschema("molecule.json")

# Access molecular properties
print(f"Charge: {system.metadata['molecular_charge']}")
print(f"Multiplicity: {system.metadata['molecular_multiplicity']}")
```

### VASP Reader

Reads POSCAR/CONTCAR files with lattice vector handling:

```python
from autofragment.io import read_poscar

system = read_poscar("POSCAR")

# Lattice vectors are in metadata
lattice = system.metadata["lattice"]
```

## Writing Fragment Files

### Quick Start

```python
from autofragment.io import write_gamess_fmo, write_psi4_sapt

# After fragmenting a system...
write_gamess_fmo(fragments, "fmo_calc.inp", basis="6-31G*")
write_psi4_sapt(fragments[:2], "sapt_dimer.dat")
```

### Supported Output Formats

| Format | Function | Description |
|--------|----------|-------------|
| GAMESS FMO | `write_gamess_fmo()` | FMO with $FMO, $FMOBND, $DATA |
| GAMESS EFMO | `write_gamess_efmo()` | Effective Fragment MO |
| GAMESS EFP | `write_gamess_efp()` | Effective Fragment Potential |
| Psi4 SAPT | `write_psi4_sapt()` | SAPT with fragment separators |
| Psi4 Fragment | `write_psi4_fragment()` | General fragment calculation |
| Q-Chem EFP | `write_qchem_efp()` | EFP with $molecule sections |
| Q-Chem XSAPT | `write_qchem_xsapt()` | Extended SAPT |
| Q-Chem FRAGMO | `write_qchem_fragmo()` | Fragment MO analysis |
| NWChem | `write_nwchem_fragment()` | Fragment calculation |
| NWChem BSSE | `write_nwchem_bsse()` | Counterpoise correction |
| ORCA | `write_orca_fragment()` | Single fragment job |
| ORCA Multi-job | `write_orca_multijob()` | Separate jobs per fragment |
| QCSchema | `write_qcschema()` | JSON with fragment annotations |
| Molpro | `write_molpro_sapt()` | SAPT calculation |
| Turbomole | `write_turbomole_fragment()` | Coord file |
| CFOUR | `write_cfour_fragment()` | ZMAT format |
| XYZ | `write_xyz_fragments()` | XYZ with fragment markers |
| PDB | `write_pdb_fragments()` | PDB with fragments as chains |

### GAMESS FMO Writer

Generates complete GAMESS input for Fragment Molecular Orbital calculations:

```python
from autofragment.io import write_gamess_fmo

write_gamess_fmo(
    fragments,
    "fmo_calc.inp",
    basis="6-31G*",
    method="RHF",
    runtype="energy",
    memory=500,  # MW
    fmo_level=2,
    nbody=2,
)
```

Output includes:
- `$CONTRL` - Control options
- `$SYSTEM` - Memory settings
- `$BASIS` - Basis set specification
- `$FMO` - Fragment definitions (NFRAG, INDAT, charges, multiplicities)
- `$GDDI` - Parallel settings
- `$DATA` - Atomic coordinates

### Psi4 SAPT Writer

Generates Psi4 input for Symmetry-Adapted Perturbation Theory:

```python
from autofragment.io import write_psi4_sapt

# SAPT requires exactly 2 fragments (dimer)
write_psi4_sapt(
    [monomer_a, monomer_b],
    "sapt_dimer.dat",
    method="sapt0",  # or sapt2, sapt2+, etc.
    basis="jun-cc-pvdz",
    freeze_core=True,
)
```

### Q-Chem EFP Writer

Generates Q-Chem input for EFP calculations:

```python
from autofragment.io import write_qchem_efp

write_qchem_efp(
    fragments,
    "efp_calc.in",
    qm_fragment_indices=[0],  # First fragment is QM
    method="hf",
    basis="6-31g*",
)
```

### PDB Fragment Writer

Writes fragments encoded in PDB format:

```python
from autofragment.io import write_pdb_fragments

# Fragments as separate chains (A, B, C, ...)
write_pdb_fragments(fragments, "output.pdb", mode="chains")

# Fragments as separate MODELs
write_pdb_fragments(fragments, "output.pdb", mode="models")

# Fragments as separate residues
write_pdb_fragments(fragments, "output.pdb", mode="residues")
```

## Format Registry

The FormatRegistry provides a unified interface for format handling:

```python
from autofragment.io import FormatRegistry

# Auto-detect format from extension
system = FormatRegistry.read("molecule.pdb")

# List supported formats
formats = FormatRegistry.supported_formats()
print(f"Readers: {formats['read']}")
print(f"Writers: {formats['write']}")

# Register custom format
def my_reader(filepath):
    # Custom parsing logic
    return ChemicalSystem(...)

FormatRegistry.register_reader(
    "myformat",
    my_reader,
    extensions=[".mfmt"],
    description="My custom format"
)
```

## Coordinate Units

Different formats use different coordinate units:

| Format | Native Units | AutoFragment Handling |
|--------|--------------|----------------------|
| PDB, MOL2, SDF | Angstroms | Direct read |
| QCSchema | Bohr | Converted to Angstroms on read |
| VASP | Angstroms or fractional | Converted to Cartesian Angstroms |
| CIF | Fractional | Converted to Cartesian Angstroms |

## Error Handling

All readers raise appropriate exceptions:

```python
from autofragment.io import read_pdb

try:
    system = read_pdb("missing.pdb")
except FileNotFoundError:
    print("File not found")

try:
    system = read_pdb("malformed.pdb")
except ValueError as e:
    print(f"Parse error: {e}")
```

## Tips for Large Files

For large molecular systems:

```python
# SDF: Process molecules one at a time
from autofragment.io.readers.sdf import read_sdf_multi

for system in read_sdf_multi("large_library.sdf"):
    process(system)

# PDB: Read only specific model
system = read_pdb("trajectory.pdb", model=100)
```

## See Also

- {doc}`python_api` - Full Python API reference
- {doc}`core_concepts` - ChemicalSystem and Fragment types
- {doc}`api/autofragment` - Auto-generated API documentation
