# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
GAMESS FMO/EFMO/EFP input file writer.

This module provides functions for generating GAMESS input files
for fragment-based calculations (FMO, EFMO, EFP methods).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autofragment.core.types import ChemicalSystem, Fragment

# Element symbol to atomic number
_ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
}

# GAMESS basis set name mapping
_GAMESS_BASIS_MAP = {
    "sto-3g": "STO NGAUSS=3",
    "3-21g": "N21 NGAUSS=3",
    "6-31g": "N31 NGAUSS=6",
    "6-31g*": "N31 NGAUSS=6 NDFUNC=1",
    "6-31g**": "N31 NGAUSS=6 NDFUNC=1 NPFUNC=1",
    "6-31+g*": "N31 NGAUSS=6 NDFUNC=1 DIFFSP=.TRUE.",
    "6-311g": "N311 NGAUSS=6",
    "6-311g*": "N311 NGAUSS=6 NDFUNC=1",
    "6-311g**": "N311 NGAUSS=6 NDFUNC=1 NPFUNC=1",
    "cc-pvdz": "CCD",
    "cc-pvtz": "CCT",
    "cc-pvqz": "CCQ",
    "aug-cc-pvdz": "ACCD",
    "aug-cc-pvtz": "ACCT",
}


def _format_gamess_basis(basis: str) -> str:
    """
    Convert basis set name to GAMESS format.

    Parameters
    ----------
    basis : str
        Basis set name (e.g., "6-31G*", "cc-pVDZ").

    Returns
    -------
    str
        GAMESS $BASIS group content.
    """
    basis_lower = basis.lower().replace(" ", "")

    if basis_lower in _GAMESS_BASIS_MAP:
        return _GAMESS_BASIS_MAP[basis_lower]

    # Try to parse as GBASIS=... format
    if "=" in basis:
        return basis

    # Default: use as GBASIS
    return f"GBASIS={basis.upper()}"


def _find_fragment_boundaries(
    fragments: List[Fragment],
    system: ChemicalSystem,
) -> List[Tuple[int, int, int, int]]:
    """
    Find broken covalent bonds between fragments.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    system : ChemicalSystem
        The chemical system with bond information.

    Returns
    -------
    list
        List of (atom1_idx, atom2_idx, frag1_idx, frag2_idx) tuples
        for each interfragment bond.
    """
    # Build atom-to-fragment mapping
    atom_to_frag: Dict[int, int] = {}

    for frag_idx, frag in enumerate(fragments):
        coords = frag.get_coords()
        for i in range(len(frag.symbols)):
            # Match atom by finding closest in system
            frag_coord = coords[i]
            for sys_idx, atom in enumerate(system.atoms):
                if np.allclose(atom.coords, frag_coord, atol=1e-3):
                    atom_to_frag[sys_idx] = frag_idx
                    break

    # Find interfragment bonds
    broken_bonds = []
    for bond in system.bonds:
        a1, a2 = bond["atom1"], bond["atom2"]
        if a1 in atom_to_frag and a2 in atom_to_frag:
            f1, f2 = atom_to_frag[a1], atom_to_frag[a2]
            if f1 != f2:
                broken_bonds.append((a1, a2, f1, f2))

    return broken_bonds


def _write_data_group(
    f,
    fragments: List[Fragment],
    title: str = "GAMESS FMO calculation",
) -> None:
    """
    Write GAMESS $DATA group with atoms from fragments.

    Parameters
    ----------
    f : file
        Open file handle.
    fragments : list
        List of Fragment objects.
    title : str
        Title for the calculation.
    """
    f.write(" $DATA\n")
    f.write(f"{title}\n")
    f.write("C1\n")  # No symmetry

    for frag in fragments:
        coords = frag.get_coords()
        for i, symbol in enumerate(frag.symbols):
            znuc = _ELEMENT_TO_Z.get(symbol.upper(), 6)
            x, y, z = coords[i]
            f.write(f"{symbol:2s}  {znuc:5.1f} {x:15.10f} {y:15.10f} {z:15.10f}\n")

    f.write(" $END\n")


def write_gamess_fmo(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "6-31G*",
    runtype: str = "energy",
    method: str = "RHF",
    memory: int = 500,
    title: str = "AutoFragment FMO calculation",
    fmo_level: int = 2,
    nbody: int = 2,
    extra_contrl: Optional[Dict[str, str]] = None,
    extra_fmo: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write GAMESS FMO input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects defining the fragmentation.
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name. Default is "6-31G*".
    runtype : str, optional
        GAMESS run type (energy, gradient, etc.). Default is "energy".
    method : str, optional
        SCF type (RHF, UHF, ROHF). Default is "RHF".
    memory : int, optional
        Memory in MW. Default is 500.
    title : str, optional
        Title for the calculation.
    fmo_level : int, optional
        FMO level (2 or 3). Default is 2.
    nbody : int, optional
        N-body expansion (2 or 3). Default is 2.
    extra_contrl : dict, optional
        Additional $CONTRL keywords.
    extra_fmo : dict, optional
        Additional $FMO keywords.

    Examples
    --------
    >>> from autofragment.core import Fragment
    >>> frags = [Fragment(id="F1", symbols=["O", "H", "H"], geometry=[0,0,0, 1,0,0, -1,0,0])]
    >>> write_gamess_fmo(frags, "water.inp")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_frags = len(fragments)

    # Count total atoms and build INDAT
    total_atoms = 0
    indat_values = []

    for frag_idx, frag in enumerate(fragments, 1):
        n_atoms = frag.n_atoms
        indat_values.extend([frag_idx] * n_atoms)
        total_atoms += n_atoms

    with open(path, "w") as f:
        # $CONTRL group
        f.write(f" $CONTRL SCFTYP={method.upper()} RUNTYP={runtype.upper()}\n")
        f.write("         COORD=UNIQUE LOCAL=NONE\n")
        if extra_contrl:
            for key, val in extra_contrl.items():
                f.write(f"         {key.upper()}={val}\n")
        f.write(" $END\n")

        # $SYSTEM group
        f.write(f" $SYSTEM MWORDS={memory} $END\n")

        # $BASIS group
        f.write(f" $BASIS {_format_gamess_basis(basis)} $END\n")

        # $FMO group
        f.write(" $FMO\n")
        f.write(f"    NFRAG={n_frags} NBODY={nbody}\n")

        # Fragment charges
        icharg_str = ",".join(str(f.molecular_charge) for f in fragments)
        f.write(f"    ICHARG(1)={icharg_str}\n")

        # Fragment multiplicities
        mult_str = ",".join(str(f.molecular_multiplicity) for f in fragments)
        f.write(f"    MULT(1)={mult_str}\n")

        # INDAT - fragment assignment for each atom
        # Format: INDAT(1)=1,1,1,2,2,2,...
        # Split into multiple lines if too long
        max_per_line = 10
        f.write("    INDAT(1)=")
        for i in range(0, len(indat_values), max_per_line):
            chunk = indat_values[i:i + max_per_line]
            if i > 0:
                f.write("             ")
            f.write(",".join(str(x) for x in chunk))
            if i + max_per_line < len(indat_values):
                f.write(",\n")
            else:
                f.write("\n")

        if extra_fmo:
            for key, val in extra_fmo.items():
                f.write(f"    {key.upper()}={val}\n")

        f.write(" $END\n")

        # $GDDI group for parallel
        f.write(" $GDDI NGROUP=1 $END\n")

        # $DATA group
        _write_data_group(f, fragments, title)


def write_gamess_efmo(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "6-31G*",
    memory: int = 500,
    title: str = "AutoFragment EFMO calculation",
    extra_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write GAMESS EFMO (Effective Fragment Molecular Orbital) input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name. Default is "6-31G*".
    memory : int, optional
        Memory in MW. Default is 500.
    title : str, optional
        Title for the calculation.
    extra_options : dict, optional
        Additional options.
    """
    # EFMO uses FMO framework with specific keywords
    write_gamess_fmo(
        fragments=fragments,
        filepath=filepath,
        basis=basis,
        memory=memory,
        title=title,
        extra_fmo={"MODGRD": "1"},  # EFMO-specific
    )


def write_gamess_efp(
    fragments: List[Fragment],
    qm_fragment_indices: List[int],
    filepath: Union[str, Path],
    efp_potentials: Optional[Dict[str, str]] = None,
    basis: str = "6-31G*",
    method: str = "RHF",
    memory: int = 500,
    title: str = "AutoFragment EFP calculation",
) -> None:
    """
    Write GAMESS EFP (Effective Fragment Potential) input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    qm_fragment_indices : list
        Indices of fragments to treat at QM level (0-indexed).
    filepath : str or Path
        Output file path.
    efp_potentials : dict, optional
        Mapping of fragment type to EFP potential name.
    basis : str, optional
        Basis set for QM region. Default is "6-31G*".
    method : str, optional
        QM method. Default is "RHF".
    memory : int, optional
        Memory in MW. Default is 500.
    title : str, optional
        Title for the calculation.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    qm_frags = [fragments[i] for i in qm_fragment_indices]
    efp_frags = [f for i, f in enumerate(fragments) if i not in qm_fragment_indices]

    with open(path, "w") as f:
        # $CONTRL
        f.write(f" $CONTRL SCFTYP={method.upper()} RUNTYP=ENERGY\n")
        f.write("         COORD=UNIQUE LOCAL=NONE\n")
        f.write(" $END\n")

        # $SYSTEM
        f.write(f" $SYSTEM MWORDS={memory} $END\n")

        # $BASIS
        f.write(f" $BASIS {_format_gamess_basis(basis)} $END\n")

        # $EFRAG for EFP fragments
        if efp_frags:
            f.write(" $EFRAG\n")
            f.write(f"    NFRAGS={len(efp_frags)}\n")
            for i, frag in enumerate(efp_frags, 1):
                pot_name = "WATER" if efp_potentials is None else efp_potentials.get(frag.id, "WATER")
                f.write(f"    FRAGNAME({i})={pot_name}\n")
            f.write(" $END\n")

            # EFP coordinates
            f.write(" $EFCOOR\n")
            for frag in efp_frags:
                coords = frag.get_coords()
                for i, symbol in enumerate(frag.symbols):
                    x, y, z = coords[i]
                    f.write(f" {symbol:2s}  {x:12.8f} {y:12.8f} {z:12.8f}\n")
            f.write(" $END\n")

        # $DATA for QM fragments
        _write_data_group(f, qm_frags, title)
