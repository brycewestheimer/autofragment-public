# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Molecular structure file readers for AutoFragment.

This module provides readers for various molecular file formats:
- PDB: Protein Data Bank format with CONECT records
- MOL2: Tripos MOL2 format with atom types
- SDF: MDL Structure Data File (V2000 and V3000)
- QCSchema: MolSSI standard JSON format
- GAMESS: GAMESS-US input file format
- Psi4: Psi4 input file format
- Q-Chem: Q-Chem input file format
- ORCA: ORCA input file format
- VASP: VASP POSCAR/CONTCAR format
- CIF: Crystallographic Information File
- NWChem: NWChem input file format
"""

from autofragment.io.readers.cif import read_cif
from autofragment.io.readers.mol2 import read_mol2
from autofragment.io.readers.pdb import iter_pdb_atoms, read_pdb, read_pdb_lazy
from autofragment.io.readers.qc_inputs import (
    read_gamess_input,
    read_nwchem_input,
    read_orca_input,
    read_psi4_input,
    read_qchem_input,
)
from autofragment.io.readers.qcschema import qcschema_to_system, read_qcschema
from autofragment.io.readers.sdf import read_sdf, read_sdf_multi
from autofragment.io.readers.vasp import read_contcar, read_poscar

__all__ = [
    # PDB
    "read_pdb",
    "iter_pdb_atoms",
    "read_pdb_lazy",
    # MOL2
    "read_mol2",
    # SDF/MOL
    "read_sdf",
    "read_sdf_multi",
    # QCSchema
    "read_qcschema",
    "qcschema_to_system",
    # QC program inputs
    "read_gamess_input",
    "read_psi4_input",
    "read_qchem_input",
    "read_orca_input",
    "read_nwchem_input",
    # VASP
    "read_poscar",
    "read_contcar",
    # CIF
    "read_cif",
]
