# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
CFOUR fragment input file writer.

This module provides functions for generating CFOUR input files
for fragment-based calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from autofragment.core.types import Fragment


def write_cfour_fragment(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "CCSD(T)",
    basis: str = "PVDZ",
    memory: int = 2000,
    title: str = "AutoFragment CFOUR calculation",
) -> None:
    """
    Write CFOUR ZMAT input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path (typically "ZMAT").
    method : str, optional
        Calculation method. Default is "CCSD(T)".
    basis : str, optional
        Basis set name. Default is "PVDZ".
    memory : int, optional
        Memory in MB. Default is 2000.
    title : str, optional
        Title comment.

    Notes
    -----
    CFOUR uses ZMAT format with Cartesian coordinates.
    Basis set names follow CFOUR conventions (PVDZ, PVTZ, etc.).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_charge = sum(f.molecular_charge for f in fragments)
    total_mult = max(f.molecular_multiplicity for f in fragments)

    with open(path, "w") as f:
        # Title line
        f.write(f"{title}\n")

        # Coordinates
        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"{symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        # Blank line separates geometry from keywords
        f.write("\n")

        # Keyword section
        f.write(f"*CFOUR(CALC_LEVEL={method}\n")
        f.write(f"BASIS={basis}\n")
        f.write(f"CHARGE={total_charge}\n")
        f.write(f"MULTIPLICITY={total_mult}\n")
        f.write(f"MEMORY_SIZE={memory}\n")
        f.write("COORDINATES=CARTESIAN\n")
        f.write("UNITS=ANGSTROM\n")
        f.write("SCF_CONV=9\n")
        f.write("CC_CONV=9)\n")
