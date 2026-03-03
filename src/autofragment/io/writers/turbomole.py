# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Turbomole fragment input file writer.

This module provides functions for generating Turbomole input files
for fragment-based calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from autofragment.core.types import Fragment

# Conversion: Angstrom to Bohr (Turbomole uses Bohr by default)
_ANGSTROM_TO_BOHR = 1.8897259886


def write_turbomole_fragment(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "def2-SVP",
    method: str = "dft",
    functional: str = "b3-lyp",
    use_bohr: bool = True,
    title: str = "AutoFragment Turbomole calculation",
) -> None:
    """
    Write Turbomole coord file for fragment calculation.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path (typically "coord").
    basis : str, optional
        Basis set name. Default is "def2-SVP".
    method : str, optional
        Method type (dft, hf). Default is "dft".
    functional : str, optional
        DFT functional if method is dft. Default is "b3-lyp".
    use_bohr : bool, optional
        Write coordinates in Bohr. Default is True.
    title : str, optional
        Title comment.

    Notes
    -----
    This writes the coord file. Additional control files (control, basis, etc.)
    should be generated with define or manually.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    scale = _ANGSTROM_TO_BOHR if use_bohr else 1.0

    with open(path, "w") as f:
        f.write("$coord\n")

        atom_count = 0
        for frag_idx, frag in enumerate(fragments):
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i] * scale
                # Turbomole format: x y z element
                f.write(f"  {x:18.14f}  {y:18.14f}  {z:18.14f}  {symbol.lower()}\n")
                atom_count += 1

        f.write("$end\n")

    # Write a simple control file hint
    control_path = path.parent / "control.hint"
    with open(control_path, "w") as f:
        f.write(f"# Turbomole control file hints for {title}\n")
        f.write("# Run 'define' to generate proper control file\n\n")
        f.write("# Suggested settings:\n")
        f.write(f"# - Basis: {basis}\n")
        f.write(f"# - Method: {method}\n")
        if method.lower() == "dft":
            f.write(f"# - Functional: {functional}\n")
        f.write(f"# - Charge: {sum(f.molecular_charge for f in fragments)}\n")
        f.write(f"# - Atoms: {atom_count}\n")
        f.write(f"# - Fragments: {len(fragments)}\n")

        # Fragment info
        f.write("\n# Fragment information:\n")
        atom_start = 1
        for frag in fragments:
            atom_end = atom_start + frag.n_atoms - 1
            f.write(f"# - {frag.id}: atoms {atom_start}-{atom_end}, charge={frag.molecular_charge}\n")
            atom_start = atom_end + 1
