# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Q-Chem EFP/XSAPT/FRAGMO input file writer.

This module provides functions for generating Q-Chem input files
for fragment-based calculations (EFP, XSAPT, FRAGMO methods).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from autofragment.core.types import Fragment


def write_qchem_efp(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "hf",
    basis: str = "6-31g*",
    qm_fragment_indices: Optional[List[int]] = None,
    title: str = "AutoFragment EFP calculation",
    extra_rem: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write Q-Chem EFP input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method for QM region. Default is "hf".
    basis : str, optional
        Basis set name. Default is "6-31g*".
    qm_fragment_indices : list, optional
        Indices of fragments to treat at QM level (0-indexed).
        If None, first fragment is QM. Default is None.
    title : str, optional
        Title comment.
    extra_rem : dict, optional
        Additional $rem keywords.

    Examples
    --------
    >>> from autofragment.core import Fragment
    >>> frags = [
    ...     Fragment(id="QM", symbols=["O", "H", "H"], geometry=[0,0,0, 1,0,0, -1,0,0]),
    ...     Fragment(id="EFP", symbols=["O", "H", "H"], geometry=[3,0,0, 4,0,0, 2,0,0])
    ... ]
    >>> write_qchem_efp(frags, "qm_efp.in", qm_fragment_indices=[0])
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if qm_fragment_indices is None:
        qm_fragment_indices = [0] if fragments else []

    qm_frags = [fragments[i] for i in qm_fragment_indices]
    efp_indices = [i for i in range(len(fragments)) if i not in qm_fragment_indices]
    efp_frags = [fragments[i] for i in efp_indices]

    # Calculate total charge and multiplicity for QM region
    total_charge = sum(f.molecular_charge for f in qm_frags)
    total_mult = max((f.molecular_multiplicity for f in qm_frags), default=1)

    with open(path, "w") as f:
        # Comment
        f.write(f"$comment\n{title}\n$end\n\n")

        # QM molecule section
        f.write("$molecule\n")
        f.write(f"{total_charge} {total_mult}\n")

        for frag in qm_frags:
            coords = frag.get_coords()
            for i, symbol in enumerate(frag.symbols):
                x, y, z = coords[i]
                f.write(f"{symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("$end\n\n")

        # $rem section
        f.write("$rem\n")
        f.write(f"METHOD = {method.upper()}\n")
        f.write(f"BASIS = {basis}\n")

        if efp_frags:
            f.write("EFP = TRUE\n")

        if extra_rem:
            for key, val in extra_rem.items():
                f.write(f"{key.upper()} = {val}\n")

        f.write("$end\n\n")

        # EFP section for non-QM fragments
        if efp_frags:
            f.write("$efp_fragments\n")
            for frag in efp_frags:
                # Assume water for simplicity, or use fragment ID
                efp_name = frag.id.lower() if frag.id else "water"

                coords = frag.get_coords()
                if len(coords) >= 3:
                    # Use center, first atom, and reference axis
                    center = coords.mean(axis=0)
                    f.write(f"{efp_name}\n")
                    f.write(f"{center[0]:15.10f} {center[1]:15.10f} {center[2]:15.10f}\n")
            f.write("$end\n")


def write_qchem_xsapt(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    basis: str = "aug-cc-pvdz",
    title: str = "AutoFragment XSAPT calculation",
    xsapt_level: str = "xsapt(ksdft)",
    extra_rem: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write Q-Chem XSAPT input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects (typically 2 for SAPT).
    filepath : str or Path
        Output file path.
    basis : str, optional
        Basis set name. Default is "aug-cc-pvdz".
    title : str, optional
        Title comment.
    xsapt_level : str, optional
        XSAPT level (xsapt, xsapt(ksdft), etc.). Default is "xsapt(ksdft)".
    extra_rem : dict, optional
        Additional $rem keywords.
    """
    if len(fragments) < 2:
        raise ValueError("XSAPT requires at least 2 fragments")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Total charge and multiplicity
    total_charge = sum(f.molecular_charge for f in fragments)
    total_mult = max(f.molecular_multiplicity for f in fragments)

    with open(path, "w") as f:
        f.write(f"$comment\n{title}\n$end\n\n")

        # Molecule with fragment markers
        f.write("$molecule\n")
        f.write(f"{total_charge} {total_mult}\n")

        for i, frag in enumerate(fragments):
            if i > 0:
                f.write("--\n")
            f.write(f"{frag.molecular_charge} {frag.molecular_multiplicity}\n")

            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"{symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("$end\n\n")

        # $rem
        f.write("$rem\n")
        f.write("JOBTYPE = sp\n")
        f.write(f"BASIS = {basis}\n")
        f.write("EXCHANGE = gen\n")
        f.write("XSAPT = TRUE\n")

        if "ksdft" in xsapt_level.lower():
            f.write("DFT_D = D3\n")

        if extra_rem:
            for key, val in extra_rem.items():
                f.write(f"{key.upper()} = {val}\n")

        f.write("$end\n")


def write_qchem_fragmo(
    fragments: List[Fragment],
    filepath: Union[str, Path],
    method: str = "hf",
    basis: str = "6-31g*",
    title: str = "AutoFragment FRAGMO calculation",
    extra_rem: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write Q-Chem FRAGMO (fragment molecular orbital) input file.

    Parameters
    ----------
    fragments : list
        List of Fragment objects.
    filepath : str or Path
        Output file path.
    method : str, optional
        QM method. Default is "hf".
    basis : str, optional
        Basis set name. Default is "6-31g*".
    title : str, optional
        Title comment.
    extra_rem : dict, optional
        Additional $rem keywords.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_charge = sum(f.molecular_charge for f in fragments)
    total_mult = max(f.molecular_multiplicity for f in fragments)

    with open(path, "w") as f:
        f.write(f"$comment\n{title}\n$end\n\n")

        # Molecule
        f.write("$molecule\n")
        f.write(f"{total_charge} {total_mult}\n")

        for i, frag in enumerate(fragments):
            if i > 0:
                f.write("--\n")
            f.write(f"{frag.molecular_charge} {frag.molecular_multiplicity}\n")

            coords = frag.get_coords()
            for j, symbol in enumerate(frag.symbols):
                x, y, z = coords[j]
                f.write(f"{symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")

        f.write("$end\n\n")

        # $rem
        f.write("$rem\n")
        f.write(f"METHOD = {method.upper()}\n")
        f.write(f"BASIS = {basis}\n")
        f.write("FRAGMO = TRUE\n")
        f.write(f"NFRAGMO = {len(fragments)}\n")

        if extra_rem:
            for key, val in extra_rem.items():
                f.write(f"{key.upper()} = {val}\n")

        f.write("$end\n")
