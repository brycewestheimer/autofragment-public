# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantum chemistry program output file writers for AutoFragment.

This module provides writers for generating input files for various QC programs
with fragment annotations:
- GAMESS: FMO, EFMO, EFP methods
- Psi4: Fragment, SAPT methods
- Q-Chem: EFP, XSAPT, FRAGMO methods
- NWChem: Fragment, BSSE methods
- ORCA: Multi-job fragment calculations
- QCSchema: JSON format with fragment annotations
- Molpro: SAPT, DF-LMP2 methods
- Turbomole: Fragment calculations
- CFOUR: Fragment calculations
- Generic XYZ: With fragment markers
- PDB: With fragments as chains/models
"""

from autofragment.io.writers.cfour import (
    write_cfour_fragment,
)
from autofragment.io.writers.gamess import (
    write_gamess_efmo,
    write_gamess_efp,
    write_gamess_fmo,
)
from autofragment.io.writers.molpro import (
    write_molpro_sapt,
)
from autofragment.io.writers.nwchem import (
    write_nwchem_bsse,
    write_nwchem_fragment,
)
from autofragment.io.writers.orca import (
    write_orca_fragment,
    write_orca_multijob,
)
from autofragment.io.writers.pdb_writer import (
    write_pdb_fragments,
)
from autofragment.io.writers.psi4 import (
    write_psi4_fragment,
    write_psi4_sapt,
)
from autofragment.io.writers.qchem import (
    write_qchem_efp,
    write_qchem_fragmo,
    write_qchem_xsapt,
)
from autofragment.io.writers.qcschema_writer import (
    system_to_qcschema,
    write_qcmanybody_input,
    write_qcschema,
)
from autofragment.io.writers.registry import FormatRegistry
from autofragment.io.writers.turbomole import (
    write_turbomole_fragment,
)
from autofragment.io.writers.xyz_writer import (
    write_xyz_fragments,
)

__all__ = [
    # GAMESS
    "write_gamess_fmo",
    "write_gamess_efmo",
    "write_gamess_efp",
    # Psi4
    "write_psi4_sapt",
    "write_psi4_fragment",
    # Q-Chem
    "write_qchem_efp",
    "write_qchem_xsapt",
    "write_qchem_fragmo",
    # NWChem
    "write_nwchem_fragment",
    "write_nwchem_bsse",
    # ORCA
    "write_orca_fragment",
    "write_orca_multijob",
    # QCSchema
    "write_qcschema",
    "write_qcmanybody_input",
    "system_to_qcschema",
    # Molpro
    "write_molpro_sapt",
    # Turbomole
    "write_turbomole_fragment",
    # CFOUR
    "write_cfour_fragment",
    # XYZ
    "write_xyz_fragments",
    # PDB
    "write_pdb_fragments",
    # Registry
    "FormatRegistry",
]
