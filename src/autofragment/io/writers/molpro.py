# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Molpro SAPT and DF-LMP2 input file writer.

This module provides functions for generating Molpro input files
for fragment-based calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from autofragment.core.types import Fragment


def write_molpro_sapt(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "aug-cc-pvdz",
    sapt_level: str = "sapt",
    memory: int = 500,
    title: str = "AutoFragment Molpro SAPT calculation",
) -> None:
    """
    Write Molpro SAPT input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects (exactly 2 for SAPT).
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name. Default is "aug-cc-pvdz".
    sapt_level : str, optional
        SAPT level (sapt, sapt2+, etc.). Default is "sapt".
    memory : int, optional
        Memory in MW. Default is 500.
    title : str, optional
        Title comment.

    Notes
    -----
    Molpro SAPT requires exactly 2 fragments (monomers).
    """
    if len(fragments) != 2:
        raise ValueError("Molpro SAPT requires exactly 2 fragments")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Memory
        f.write(f"memory,{memory},m\n\n")

        # Title
        f.write(f"! {title}\n\n")

        # Dimer geometry
        f.write("geometry={\n")

        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"  {symbol:2s}, {x:15.10f}, {y:15.10f}, {z:15.10f}\n")

        f.write("}\n\n")

        # Basis
        f.write(f"basis={basis}\n\n")

        # Define monomers
        monomer_a = fragments[0]
        monomer_b = fragments[1]

        atom_offset_b = monomer_a.n_atoms

        # Atom numbering for monomers
        atoms_a = ",".join(str(i+1) for i in range(monomer_a.n_atoms))
        atoms_b = ",".join(str(i+atom_offset_b+1) for i in range(monomer_b.n_atoms))

        f.write(f"! Monomer A atoms: {atoms_a}\n")
        f.write(f"! Monomer B atoms: {atoms_b}\n\n")

        # SAPT calculation
        f.write("! Dimer HF and localization\n")
        f.write("{{hf}}\n\n")

        f.write(f"! {sapt_level.upper()} calculation\n")
        f.write(f"{{df-{sapt_level}}}\n")


def write_molpro_lmp2(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "cc-pvdz",
    memory: int = 500,
    title: str = "AutoFragment Molpro DF-LMP2 calculation",
) -> None:
    """
    Write Molpro DF-LMP2 input file with domain assignments.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name. Default is "cc-pvdz".
    memory : int, optional
        Memory in MW. Default is 500.
    title : str, optional
        Title comment.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"memory,{memory},m\n\n")
        f.write(f"! {title}\n\n")

        # Geometry
        f.write("geometry={\n")
        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"  {symbol:2s}, {x:15.10f}, {y:15.10f}, {z:15.10f}\n")
        f.write("}\n\n")

        f.write(f"basis={basis}\n\n")

        # HF
        f.write("{hf}\n\n")

        # DF-LMP2
        f.write("{df-lmp2}\n")
