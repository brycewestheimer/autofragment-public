# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""I/O utilities for reading and writing molecular data.

This module provides comprehensive I/O support for molecular file formats:

Readers:
- PDB: Protein Data Bank format with CONECT records
- MOL2: Tripos MOL2 format with atom types
- SDF: MDL Structure Data File (V2000/V3000)
- QCSchema: MolSSI standard JSON format
- GAMESS, Psi4, Q-Chem, ORCA, NWChem: QC program inputs
- VASP: POSCAR/CONTCAR format
- CIF: Crystallographic Information File

Writers:
- GAMESS: FMO, EFMO, EFP methods
- Psi4: Fragment, SAPT methods
- Q-Chem: EFP, XSAPT, FRAGMO methods
- NWChem, ORCA, Molpro, Turbomole, CFOUR: Fragment calculations
- QCSchema: JSON with fragment annotations
- XYZ, PDB: With fragment markers
"""

from autofragment.io.output import write_json

# Import readers
from autofragment.io.readers import (
    qcschema_to_system,
    read_cif,
    read_contcar,
    read_gamess_input,
    read_mol2,
    read_nwchem_input,
    read_orca_input,
    read_pdb,
    read_poscar,
    read_psi4_input,
    read_qchem_input,
    read_qcschema,
    read_sdf,
    read_sdf_multi,
)

# Import writers
from autofragment.io.writers import (
    FormatRegistry,
    system_to_qcschema,
    write_cfour_fragment,
    write_gamess_efmo,
    write_gamess_efp,
    write_gamess_fmo,
    write_molpro_sapt,
    write_nwchem_bsse,
    write_nwchem_fragment,
    write_orca_fragment,
    write_orca_multijob,
    write_pdb_fragments,
    write_psi4_fragment,
    write_psi4_sapt,
    write_qchem_efp,
    write_qchem_fragmo,
    write_qchem_xsapt,
    write_qcmanybody_input,
    write_qcschema,
    write_turbomole_fragment,
    write_xyz_fragments,
)
from autofragment.io.xyz import (
    ValidationError,
    read_xyz,
    read_xyz_molecules,
    write_xyz,
)

__all__ = [
    # Original exports
    "read_xyz",
    "read_xyz_molecules",
    "write_xyz",
    "ValidationError",
    "write_json",
    # Readers
    "read_pdb",
    "read_mol2",
    "read_sdf",
    "read_sdf_multi",
    "read_qcschema",
    "qcschema_to_system",
    "read_gamess_input",
    "read_psi4_input",
    "read_qchem_input",
    "read_orca_input",
    "read_nwchem_input",
    "read_poscar",
    "read_contcar",
    "read_cif",
    # Writers
    "write_gamess_fmo",
    "write_gamess_efmo",
    "write_gamess_efp",
    "write_psi4_sapt",
    "write_psi4_fragment",
    "write_qchem_efp",
    "write_qchem_xsapt",
    "write_qchem_fragmo",
    "write_nwchem_fragment",
    "write_nwchem_bsse",
    "write_orca_fragment",
    "write_orca_multijob",
    "write_qcschema",
    "write_qcmanybody_input",
    "system_to_qcschema",
    "write_molpro_sapt",
    "write_turbomole_fragment",
    "write_cfour_fragment",
    "write_xyz_fragments",
    "write_pdb_fragments",
    "FormatRegistry",
]
