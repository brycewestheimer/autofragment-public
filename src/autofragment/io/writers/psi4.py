# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Psi4 fragment and SAPT input file writer.

This module provides functions for generating Psi4 input files
for fragment-based and SAPT calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from autofragment.core.types import Fragment


def write_psi4_sapt(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "sapt0",
    basis: str = "jun-cc-pvdz",
    memory: str = "4 GB",
    title: str = "AutoFragment SAPT calculation",
    freeze_core: bool = True,
    extra_options: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write Psi4 SAPT input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects (typically 2 for SAPT).
    filepath : str or Path
        Output file path.
    method : str, optional
        SAPT method (sapt0, sapt2, sapt2+, sapt2+(3), etc.). Default is "sapt0".
    basis : str, optional
        Basis set name. Default is "jun-cc-pvdz".
    memory : str, optional
        Memory specification. Default is "4 GB".
    title : str, optional
        Title comment.
    freeze_core : bool, optional
        Whether to freeze core orbitals. Default is True.
    extra_options : dict, optional
        Additional Psi4 options to set.

    Notes
    -----
    SAPT calculations require exactly 2 fragments (monomer A and B).
    For more fragments, use F-SAPT or MBE approaches.

    Examples
    --------
    >>> from autofragment.core import Fragment
    >>> frags = [
    ...     Fragment(id="A", symbols=["O", "H", "H"], geometry=[0,0,0, 1,0,0, -1,0,0]),
    ...     Fragment(id="B", symbols=["O", "H", "H"], geometry=[3,0,0, 4,0,0, 2,0,0])
    ... ]
    >>> write_psi4_sapt(frags, "dimer.dat")
    """
    if len(fragments) < 2:
        raise ValueError("SAPT requires at least 2 fragments")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Header comment
        f.write(f"# {title}\n")
        f.write(f"# Psi4 {method.upper()} calculation\n\n")

        # Memory
        f.write(f"memory {memory}\n\n")

        # Molecule block with fragment separators
        f.write("molecule complex {\n")

        for i, frag in enumerate(fragments):
            if i > 0:
                f.write("--\n")

            # Fragment charge and multiplicity
            f.write(f"  {frag.molecular_charge} {frag.molecular_multiplicity}\n")

            # Atoms
            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("}\n\n")

        # Options
        f.write("set {\n")
        f.write(f"  basis {basis}\n")
        f.write(f"  freeze_core {'true' if freeze_core else 'false'}\n")
        f.write("  scf_type df\n")

        if extra_options:
            for key, val in extra_options.items():
                f.write(f"  {key} {val}\n")

        f.write("}\n\n")

        # Energy call
        f.write(f"energy('{method}')\n")


def write_psi4_fragment(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "scf",
    basis: str = "6-31g*",
    memory: str = "4 GB",
    title: str = "AutoFragment calculation",
    runtype: str = "energy",
    extra_options: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write Psi4 input file with fragment definitions.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method. Default is "scf".
    basis : str, optional
        Basis set name. Default is "6-31g*".
    memory : str, optional
        Memory specification. Default is "4 GB".
    title : str, optional
        Title comment.
    runtype : str, optional
        Type of calculation (energy, gradient, optimize). Default is "energy".
    extra_options : dict, optional
        Additional Psi4 options.

    Examples
    --------
    >>> from autofragment.core import Fragment
    >>> frags = [
    ...     Fragment(id="W1", symbols=["O", "H", "H"], geometry=[0,0,0, 1,0,0, -1,0,0])
    ... ]
    >>> write_psi4_fragment(frags, "water.dat")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Header
        f.write(f"# {title}\n\n")

        # Memory
        f.write(f"memory {memory}\n\n")

        # Molecule block with fragment separators
        f.write("molecule system {\n")

        if len(fragments) > 1:
            # Multiple fragments with separators
            for i, frag in enumerate(fragments):
                if i > 0:
                    f.write("--\n")
                f.write(f"  {frag.molecular_charge} {frag.molecular_multiplicity}\n")

                coords = frag.get_coords()
                for j, symbol in enumerate(frag.symbols):
                    x, y, z = coords[j]
                    f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")
        else:
            # Single fragment (or no fragments)
            if fragments:
                frag = fragments[0]
                f.write(f"  {frag.molecular_charge} {frag.molecular_multiplicity}\n")
                coords = frag.get_coords()
                for j, symbol in enumerate(frag.symbols):
                    x, y, z = coords[j]
                    f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("}\n\n")

        # Options
        f.write("set {\n")
        f.write(f"  basis {basis}\n")

        if extra_options:
            for key, val in extra_options.items():
                f.write(f"  {key} {val}\n")

        f.write("}\n\n")

        # Run command
        if runtype.lower() == "energy":
            f.write(f"energy('{method}')\n")
        elif runtype.lower() == "gradient":
            f.write(f"gradient('{method}')\n")
        elif runtype.lower() == "optimize":
            f.write(f"optimize('{method}')\n")
        else:
            f.write(f"{runtype}('{method}')\n")


def write_psi4_fsapt(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "jun-cc-pvdz",
    memory: str = "4 GB",
    title: str = "AutoFragment F-SAPT calculation",
) -> None:
    """
    Write Psi4 F-SAPT (Functional-group SAPT) input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects. First 2 are monomers, rest define functional groups.
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name.
    memory : str, optional
        Memory specification.
    title : str, optional
        Title comment.
    """
    if len(fragments) < 2:
        raise ValueError("F-SAPT requires at least 2 monomer fragments")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"memory {memory}\n\n")

        # Molecule with 2 monomers
        f.write("molecule dimer {\n")

        for i, frag in enumerate(fragments[:2]):
            if i > 0:
                f.write("--\n")
            f.write(f"  {frag.molecular_charge} {frag.molecular_multiplicity}\n")

            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("}\n\n")

        f.write("set {\n")
        f.write(f"  basis {basis}\n")
        f.write("  scf_type df\n")
        f.write("}\n\n")

        # F-SAPT specific
        f.write("energy('fisapt0')\n")
