# Output Formats Guide

This guide covers all supported output formats for exporting fragments to quantum chemistry programs.

## Supported Programs

| Program | Format | Features |
|---------|--------|----------|
| GAMESS | FMO/EFMO/EFP | Full fragment support |
| Psi4 | SAPT/MBE | Interaction energies |
| Q-Chem | EFP/XSAPT | Embedding |
| ORCA | Multi-job | Fragment calculations |
| NWChem | BSSE | Counterpoise |
| Molpro | SAPT/LMP2 | Fragment calculations |
| Turbomole | Fragment | Coordinate export |
| CFOUR | Fragment | ZMAT export |
| Generic | XYZ/PDB | Structure only |

## GAMESS

### Standard FMO

```python
from autofragment.io import write_gamess_fmo

write_gamess_fmo(
    fragments,
    "output.inp",
    method="MP2",
    basis="6-31G*",
    runtyp="ENERGY",
    memory=800
)
```

### EFMO (Effective Fragment Molecular Orbital)

```python
from autofragment.io import write_gamess_efmo

write_gamess_efmo(
    fragments,
    "efmo.inp",
    method="MP2",
    basis="6-31G*"
)
```

### FMO Options

| Option | Description |
|--------|-------------|
| `nbody` | MBE order (2 or 3) |
| `resppc` | ESP pair approximation |
| `resdim` | Distance threshold for ESP |
| `runtyp` | ENERGY, GRADIENT, HESSIAN |

## Psi4

### Basic Output

```python
from autofragment.io import write_psi4_fragment

write_psi4_fragment(
    fragments,
    "psi4_input.dat",
    method="B3LYP",
    basis="cc-pVDZ",
    extra_options={"reference": "rhf"}
)
```

### SAPT Calculations

```python
from autofragment.io import write_psi4_sapt

write_psi4_sapt(
    [frag_a, frag_b],
    "sapt.dat",
    method="sapt2+",
    basis="aug-cc-pVDZ"
)
# Requires exactly 2 fragments (dimer)
```

### MBE in Psi4

```python
write_psi4_fragment(
    fragments,
    "mbe.dat",
    method="MP2",
    basis="cc-pVDZ",
    runtype="energy"
)
```

## Q-Chem

```python
from autofragment.io import write_qchem_efp

write_qchem_efp(
    fragments,
    "qchem.in",
    method="B3LYP",
    basis="6-31G*",
    qm_fragment_indices=[0],
)
```

### EFP in Q-Chem

```python
from autofragment.io import write_qchem_xsapt

write_qchem_xsapt(
    [frag_a, frag_b],
    "xsapt.in",
    basis="6-31G*",
    xsapt_level="xsapt(ksdft)"
)
```

## ORCA

```python
from autofragment.io import write_orca_multijob

write_orca_multijob(
    fragments,
    "orca_fragments.inp",
    method="B3LYP",
    basis="def2-SVP",
    memory=4000
)
```

### ORCA Options

```python
from autofragment.io import write_orca_fragment

write_orca_fragment(
    fragments,
    "orca_single.inp",
    method="DLPNO-CCSD(T)",
    basis="cc-pVTZ"
)
```

## NWChem

```python
from autofragment.io import write_nwchem_fragment

write_nwchem_fragment(
    fragments,
    "nwchem.nw",
    method="DFT",
    basis="6-31G*",
    extra_options={"xc": "b3lyp"}
)
```

### BSSE Counterpoise

```python
from autofragment.io import write_nwchem_bsse

write_nwchem_bsse(
    [frag_a, frag_b],
    "bsse.nw",
    basis="aug-cc-pVDZ",
    method="mp2"
)
```

## Molpro

```python
from autofragment.io import write_molpro_sapt

write_molpro_sapt([frag_a, frag_b], "molpro.com", basis="aug-cc-pVDZ")
```

## Turbomole

```python
from autofragment.io import write_turbomole_fragment

write_turbomole_fragment(fragments, "coord")
```

## CFOUR

```python
from autofragment.io import write_cfour_fragment

write_cfour_fragment(fragments, "ZMAT")
```

## Generic Formats

### XYZ

```python
from autofragment.io import write_xyz_fragments

write_xyz_fragments(fragments, "fragments.xyz")
```

### PDB

```python
from autofragment.io import write_pdb_fragments

write_pdb_fragments(fragments, "fragments.pdb", mode="chains")
```

### JSON (For serialization)

```python
from autofragment.io import write_qcschema, read_qcschema

write_qcschema(system, "fragments.json")

# Load back
system = read_qcschema("fragments.json")
```

## Format Registry

Discover available formats:

```python
from autofragment.io import FormatRegistry

formats = FormatRegistry.supported_formats()
print(formats["write"])

# Get writer function by name
FormatRegistry.write("output.inp", fragments, format="gamess_fmo", method="MP2", basis="6-31G*")
```

## Writer Options

Available options are writer-specific. For example:

```python
from autofragment.io import write_gamess_fmo

write_gamess_fmo(
    fragments,
    "fmo.inp",
    method="MP2",
    basis="6-31G*",
    title="My calc",    # Title/comment
    memory=800,          # Memory request (MW)
)
```

## Batch Writing

Write multiple fragments at once:

```python
from autofragment.io import write_orca_fragment

for i, fragment in enumerate(fragments):
    write_orca_fragment([fragment], f"outputs/frag_{i}.inp")
```

## Custom Writers

Create custom output format:

```python
from pathlib import Path
from autofragment.io import FormatRegistry

def write_plain_xyz_fragments(fragments, filepath):
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for fragment in fragments:
            handle.write(f"# {fragment.id}\n")
            coords = fragment.get_coords()
            for i, symbol in enumerate(fragment.symbols):
                x, y, z = coords[i]
                handle.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

FormatRegistry.register_writer("plain_xyz_fragments", write_plain_xyz_fragments, extensions=[".pxyz"])
```
