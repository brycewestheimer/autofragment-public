# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
NWChem fragment and BSSE input file writer.

This module provides functions for generating NWChem input files
for fragment-based and BSSE calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from autofragment.core.types import Fragment


def write_nwchem_fragment(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "scf",
    basis: str = "6-31g*",
    memory: int = 2000,
    title: str = "AutoFragment NWChem calculation",
    extra_options: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write NWChem input file with fragment definitions.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method (scf, dft, mp2, ccsd). Default is "scf".
    basis : str, optional
        Basis set name. Default is "6-31g*".
    memory : int, optional
        Memory in MB. Default is 2000.
    title : str, optional
        Title for the calculation.
    extra_options : dict, optional
        Additional NWChem options.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_charge = sum(f.molecular_charge for f in fragments)

    with open(path, "w") as f:
        # Title
        f.write(f"title \"{title}\"\n\n")

        # Memory
        f.write(f"memory {memory} mb\n\n")

        # Charge
        f.write(f"charge {total_charge}\n\n")

        # Geometry
        f.write("geometry units angstroms noautoz\n")

        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("end\n\n")

        # Basis
        f.write("basis\n")
        f.write(f"  * library {basis}\n")
        f.write("end\n\n")

        # Method-specific blocks
        if method.lower() == "dft":
            f.write("dft\n")
            f.write("  xc b3lyp\n")
            f.write("end\n\n")
        elif method.lower() == "mp2":
            f.write("mp2\n")
            f.write("  freeze core\n")
            f.write("end\n\n")
        elif method.lower() == "ccsd":
            f.write("ccsd\n")
            f.write("  freeze core\n")
            f.write("end\n\n")

        # Task
        f.write(f"task {method.lower()} energy\n")


def write_nwchem_bsse(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "scf",
    basis: str = "6-31g*",
    memory: int = 2000,
    title: str = "AutoFragment BSSE calculation",
) -> None:
    """
    Write NWChem BSSE (counterpoise correction) input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects (typically 2 for dimer BSSE).
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method. Default is "scf".
    basis : str, optional
        Basis set name. Default is "6-31g*".
    memory : int, optional
        Memory in MB. Default is 2000.
    title : str, optional
        Title for the calculation.

    Notes
    -----
    This generates a multi-task input for counterpoise correction:
    1. Dimer calculation
    2. Monomer A in dimer basis
    3. Monomer B in dimer basis
    4. Monomer A in monomer basis
    5. Monomer B in monomer basis
    """
    if len(fragments) < 2:
        raise ValueError("BSSE calculation requires at least 2 fragments")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"title \"{title}\"\n\n")
        f.write(f"memory {memory} mb\n\n")

        # Collect all atoms
        all_atoms = []
        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                all_atoms.append({
                    "symbol": symbol,
                    "coords": coords[i],
                    "frag": frag.id,
                })

        # Calculate fragment atom ranges
        frag_ranges = []
        start = 0
        for frag in fragments:
            end = start + frag.n_atoms
            frag_ranges.append((start, end))
            start = end

        # Full dimer geometry with bqzone for ghost atoms
        f.write("# Full dimer calculation\n")
        f.write("geometry units angstroms noautoz\n")
        for atom in all_atoms:
            x, y, z = atom["coords"]
            f.write(f"  {atom['symbol']:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")
        f.write("end\n\n")

        total_charge = sum(frag.molecular_charge for frag in fragments)
        f.write(f"charge {total_charge}\n\n")

        f.write("basis\n")
        f.write(f"  * library {basis}\n")
        f.write("end\n\n")

        f.write(f"task {method.lower()} energy\n\n")

        # Monomer calculations with ghost atoms (counterpoise)
        for frag_idx, frag in enumerate(fragments):
            f.write(f"# Monomer {frag.id} in dimer basis (counterpoise)\n")
            f.write("geometry units angstroms noautoz\n")

            for i, atom in enumerate(all_atoms):
                x, y, z = atom["coords"]
                frag_start, frag_end = frag_ranges[frag_idx]

                if frag_start <= i < frag_end:
                    # Real atom
                    f.write(f"  {atom['symbol']:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")
                else:
                    # Ghost atom
                    f.write(f"  bq{atom['symbol'].lower()}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

            f.write("end\n\n")
            f.write(f"charge {frag.molecular_charge}\n\n")
            f.write(f"task {method.lower()} energy\n\n")
