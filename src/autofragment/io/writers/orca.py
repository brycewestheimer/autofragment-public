# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
ORCA fragment and multi-job input file writer.

This module provides functions for generating ORCA input files
for fragment-based calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from autofragment.core.types import Fragment


def write_orca_fragment(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "HF",
    basis: str = "def2-SVP",
    runtype: str = "SP",
    memory: int = 4000,
    nprocs: int = 1,
    title: str = "AutoFragment ORCA calculation",
    extra_keywords: Optional[List[str]] = None,
    extra_blocks: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write ORCA input file with all fragments combined.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method. Default is "HF".
    basis : str, optional
        Basis set name. Default is "def2-SVP".
    runtype : str, optional
        Run type (SP, Opt, Freq). Default is "SP".
    memory : int, optional
        Memory per core in MB. Default is 4000.
    nprocs : int, optional
        Number of processors. Default is 1.
    title : str, optional
        Title comment.
    extra_keywords : list, optional
        Additional keywords for simple input line.
    extra_blocks : dict, optional
        Additional input blocks.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_charge = sum(f.molecular_charge for f in fragments)
    total_mult = max(f.molecular_multiplicity for f in fragments)

    # Build keywords
    keywords = [f"! {method} {basis}"]
    if runtype.upper() != "SP":
        keywords[0] += f" {runtype}"

    if extra_keywords:
        keywords[0] += " " + " ".join(extra_keywords)

    with open(path, "w") as f:
        # Comment
        f.write(f"# {title}\n\n")

        # Keywords line
        f.write(f"{keywords[0]}\n\n")

        # PAL block for parallel
        if nprocs > 1:
            f.write(f"%pal\n  nprocs {nprocs}\nend\n\n")

        # Memory
        f.write(f"%maxcore {memory}\n\n")

        # Extra blocks
        if extra_blocks:
            for block_name, content in extra_blocks.items():
                f.write(f"%{block_name}\n{content}end\n\n")

        # Coordinates
        f.write(f"* xyz {total_charge} {total_mult}\n")

        for frag in fragments:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("*\n")


def write_orca_multijob(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "HF",
    basis: str = "def2-SVP",
    memory: int = 4000,
    title: str = "AutoFragment ORCA multi-job calculation",
) -> None:
    """
    Write ORCA multi-job input file with separate calculations per fragment.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method. Default is "HF".
    basis : str, optional
        Basis set name. Default is "def2-SVP".
    memory : int, optional
        Memory per core in MB. Default is 4000.
    title : str, optional
        Title comment.

    Notes
    -----
    This creates separate jobs for:
    1. Each fragment individually
    2. The complete system

    Jobs are separated by $new_job keywords.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write("# Multi-job file for fragment analysis\n\n")

        # Individual fragment calculations
        for i, frag in enumerate(fragments):
            if i > 0:
                f.write("\n$new_job\n\n")

            f.write(f"! {method} {basis}\n\n")
            f.write(f"%maxcore {memory}\n\n")

            f.write(f"# Fragment {frag.id}\n")
            f.write(f"* xyz {frag.molecular_charge} {frag.molecular_multiplicity}\n")

            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

            f.write("*\n")

        # Full system calculation
        f.write("\n$new_job\n\n")
        f.write(f"! {method} {basis}\n\n")
        f.write(f"%maxcore {memory}\n\n")

        total_charge = sum(frag.molecular_charge for frag in fragments)
        total_mult = max(frag.molecular_multiplicity for frag in fragments)

        f.write("# Full system\n")
        f.write(f"* xyz {total_charge} {total_mult}\n")

        for frag in fragments:
            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"  {symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("*\n")
