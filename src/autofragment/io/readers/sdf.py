# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
SDF/MOL file reader (MDL format).

This module provides functions for reading MDL Structure-Data Files (SDF) and MOL files
into ChemicalSystem objects, supporting both V2000 and V3000 formats.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem

# Periodic table for atomic number to element conversion
_ATOMIC_NUMBER_TO_ELEMENT = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
}


def _parse_v2000_atom(line: str) -> Dict[str, Any]:
    """
    Parse V2000 atom block line.

    Format (fixed-width columns):
    xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee

    - x,y,z: coordinates
    - aaa: atom symbol
    - dd: mass difference
    - ccc: charge
    - sss: stereo parity

    Parameters
    ----------
    line : str
        A line from the V2000 atom block.

    Returns
    -------
    dict
        Parsed atom data.
    """
    try:
        x = float(line[0:10])
        y = float(line[10:20])
        z = float(line[20:30])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid V2000 atom coordinates: {line!r}") from e

    # Atom symbol (3 chars, right-justified or left-justified)
    element = line[31:34].strip() if len(line) > 34 else "C"

    # Mass difference (relative to standard isotope)
    try:
        mass_diff = int(line[34:36]) if len(line) > 36 else 0
    except ValueError:
        warnings.warn(
            f"Invalid V2000 mass difference field; defaulting to 0: {line!r}",
            UserWarning,
            stacklevel=2,
        )
        mass_diff = 0

    # Charge (encoded: 0=0, 1=+3, 2=+2, 3=+1, 4=doublet, 5=-1, 6=-2, 7=-3)
    charge = 0.0
    if len(line) > 38:
        try:
            charge_code = int(line[36:39])
            charge_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
            charge = float(charge_map.get(charge_code, 0))
        except ValueError:
            warnings.warn(
                f"Invalid V2000 charge code field; defaulting to neutral: {line!r}",
                UserWarning,
                stacklevel=2,
            )

    return {
        "x": x,
        "y": y,
        "z": z,
        "element": element,
        "charge": charge,
        "mass_diff": mass_diff,
    }


def _parse_v2000_bond(line: str) -> Tuple[int, int, float]:
    """
    Parse V2000 bond block line.

    Format: 111222tttsssxxxrrrccc
    - 111: first atom number
    - 222: second atom number
    - ttt: bond type (1=single, 2=double, 3=triple, 4=aromatic)
    - sss: bond stereo

    Parameters
    ----------
    line : str
        A line from the V2000 bond block.

    Returns
    -------
    tuple
        (atom1_index, atom2_index, bond_order) with 0-based indices.
    """
    try:
        atom1 = int(line[0:3])
        atom2 = int(line[3:6])
        bond_type = int(line[6:9])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid V2000 bond line: {line!r}") from e

    # Bond type to order mapping
    order_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}  # 4 = aromatic
    order = order_map.get(bond_type, 1.0)

    # Convert to 0-based indices
    return (atom1 - 1, atom2 - 1, order)


def _read_v2000_block(
    lines: List[str],
    start_idx: int,
    n_atoms: int,
    n_bonds: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, float]], Dict[int, int]]:
    """
    Parse V2000 atom and bond blocks.

    Parameters
    ----------
    lines : list
        All lines of the file.
    start_idx : int
        Index of first atom line.
    n_atoms : int
        Number of atoms.
    n_bonds : int
        Number of bonds.

    Returns
    -------
    tuple
        (atoms, bonds, charge_dict) where charge_dict maps atom idx to formal charge.
    """
    atoms = []
    bonds = []
    charge_dict: Dict[int, int] = {}

    # Parse atoms
    for i in range(n_atoms):
        line = lines[start_idx + i]
        atoms.append(_parse_v2000_atom(line))

    # Parse bonds
    bond_start = start_idx + n_atoms
    for i in range(n_bonds):
        if bond_start + i >= len(lines):
            raise ValueError(
                f"V2000 bond block truncated: expected {n_bonds} bonds, got {i}"
            )
        line = lines[bond_start + i]
        try:
            bonds.append(_parse_v2000_bond(line))
        except ValueError as e:
            raise ValueError(
                f"Malformed V2000 bond record at line {bond_start + i + 1}: {line.strip()!r}"
            ) from e

    # Parse property block for charges (M  CHG)
    prop_start = bond_start + n_bonds
    for i in range(prop_start, len(lines)):
        line = lines[i]
        if line.startswith("M  END"):
            break
        if line.startswith("$$$$"):
            break

        if line.startswith("M  CHG"):
            # Format: M  CHG  n  aa1 vv1 aa2 vv2 ...
            parts = line[6:].split()
            if len(parts) >= 3:
                try:
                    n_charges = int(parts[0])
                    for j in range(n_charges):
                        atom_idx = int(parts[1 + 2*j]) - 1
                        charge_val = int(parts[2 + 2*j])
                        charge_dict[atom_idx] = charge_val
                except (ValueError, IndexError):
                    warnings.warn(
                        f"Skipping malformed M  CHG property line: {line.strip()!r}",
                        UserWarning,
                        stacklevel=2,
                    )

        elif line.startswith("M  RAD"):
            # Radical electrons (M  RAD) - just note for metadata
            pass

    return atoms, bonds, charge_dict


def _parse_v3000_block(lines: List[str], start_idx: int) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, float]]]:
    """
    Parse V3000 CTAB block.

    V3000 uses keyword-value format instead of fixed columns.

    Parameters
    ----------
    lines : list
        All lines of the file.
    start_idx : int
        Index of line after "M  V30 BEGIN CTAB".

    Returns
    -------
    tuple
        (atoms, bonds)
    """
    atoms = []
    bonds = []
    in_atom_block = False
    in_bond_block = False

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()

        if "M  V30 END CTAB" in line:
            break

        if "M  V30 BEGIN ATOM" in line:
            in_atom_block = True
            continue
        elif "M  V30 END ATOM" in line:
            in_atom_block = False
            continue
        elif "M  V30 BEGIN BOND" in line:
            in_bond_block = True
            continue
        elif "M  V30 END BOND" in line:
            in_bond_block = False
            continue

        if not line.startswith("M  V30"):
            continue

        # Remove "M  V30 " prefix
        content = line[7:].strip()

        if in_atom_block:
            # Format: index type x y z aamap [properties...]
            parts = content.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed V3000 atom record: {content!r}")
            try:
                # idx = int(parts[0])
                element = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])

                # Parse optional charge from CHG=n
                charge = 0.0
                for p in parts[6:]:
                    if p.startswith("CHG="):
                        charge = float(p[4:])
                        break

                atoms.append({
                    "x": x, "y": y, "z": z,
                    "element": element,
                    "charge": charge,
                })
            except ValueError as e:
                raise ValueError(f"Malformed V3000 atom record: {content!r}") from e

        elif in_bond_block:
            # Format: index type atom1 atom2 [properties...]
            parts = content.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed V3000 bond record: {content!r}")
            try:
                # idx = int(parts[0])
                bond_type = int(parts[1])
                atom1 = int(parts[2]) - 1
                atom2 = int(parts[3]) - 1

                order_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
                order = order_map.get(bond_type, 1.0)

                bonds.append((atom1, atom2, order))
            except ValueError as e:
                raise ValueError(f"Malformed V3000 bond record: {content!r}") from e

    return atoms, bonds


def read_sdf(
    filepath: Union[str, Path],
    molecule_index: int = 0,
) -> ChemicalSystem:
    """
    Read molecule from SDF file.

    Parameters
    ----------
    filepath : str or Path
        Path to SDF file.
    molecule_index : int, optional
        Which molecule to read (0-indexed, SDF can have many). Default is 0.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms and bonds.

    Examples
    --------
    >>> system = read_sdf("molecules.sdf")
    >>> print(f"Read {system.n_atoms} atoms")
    """
    molecules = list(read_sdf_multi(filepath))

    if not molecules:
        raise ValueError(f"No molecules found in SDF file: {filepath}")

    if molecule_index >= len(molecules):
        raise IndexError(
            f"Molecule index {molecule_index} out of range "
            f"(file contains {len(molecules)} molecules)"
        )

    return molecules[molecule_index]


def read_sdf_multi(filepath: Union[str, Path]) -> Iterator[ChemicalSystem]:
    """
    Read all molecules from SDF file.

    Parameters
    ----------
    filepath : str or Path
        Path to SDF file.

    Yields
    ------
    ChemicalSystem
        ChemicalSystem for each molecule in the file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SDF file not found: {filepath}")

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    mol_count = 0

    while i < len(lines):
        # Skip to next molecule or end
        mol = _read_mol_block(lines, i, str(path.name), mol_count)
        if mol is None:
            break

        yield mol[0]
        i = mol[1]  # Next line after this molecule
        mol_count += 1


def _read_mol_block(
    lines: List[str],
    start_idx: int,
    source_file: str,
    mol_index: int,
) -> Optional[Tuple[ChemicalSystem, int]]:
    """
    Read a single MOL block from lines.

    Parameters
    ----------
    lines : list
        All lines of the file.
    start_idx : int
        Starting line index.
    source_file : str
        Source filename for metadata.
    mol_index : int
        Molecule index for metadata.

    Returns
    -------
    tuple or None
        (ChemicalSystem, next_line_index) or None if no more molecules.
    """
    if start_idx >= len(lines):
        return None

    # Header: 3 lines (name, program/timestamp, comment)
    name = lines[start_idx].strip() if start_idx < len(lines) else ""

    # Find counts line (line 4, index 3)
    counts_idx = start_idx + 3
    if counts_idx >= len(lines):
        return None

    counts_line = lines[counts_idx]

    # Check for empty or delimiter
    if not counts_line.strip() or counts_line.strip() == "$$$$":
        # Skip to after $$$$
        for i in range(start_idx, len(lines)):
            if lines[i].strip() == "$$$$":
                return _read_mol_block(lines, i + 1, source_file, mol_index)
        return None

    # Check format version
    is_v3000 = "V3000" in counts_line.upper()

    if is_v3000:
        # Find V3000 CTAB block
        ctab_start = None
        for i in range(counts_idx, min(counts_idx + 10, len(lines))):
            if "M  V30 BEGIN CTAB" in lines[i]:
                ctab_start = i + 1
                break

        if ctab_start is None:
            raise ValueError("Malformed V3000 molecule: missing 'M  V30 BEGIN CTAB'")

        atom_data, bond_data = _parse_v3000_block(lines, ctab_start)
        charge_dict: Dict[int, int] = {}

        # Find end of molecule
        end_idx = counts_idx
        for i in range(ctab_start, len(lines)):
            if lines[i].strip() == "$$$$":
                end_idx = i + 1
                break
            elif i == len(lines) - 1:
                end_idx = i + 1
    else:
        # V2000 format
        try:
            n_atoms = int(counts_line[0:3])
            n_bonds = int(counts_line[3:6])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid V2000 counts line at line {counts_idx + 1}: {counts_line.strip()!r}"
            ) from e

        atom_start = counts_idx + 1
        atom_data, bond_data, charge_dict = _read_v2000_block(
            lines, atom_start, n_atoms, n_bonds
        )

        # Find end of molecule
        end_idx = atom_start + n_atoms + n_bonds
        for i in range(end_idx, len(lines)):
            if lines[i].strip() == "$$$$":
                end_idx = i + 1
                break
            elif lines[i].startswith("M  END"):
                # Continue to find $$$$
                continue
            elif i == len(lines) - 1:
                end_idx = i + 1

    # Apply charge overrides
    for atom_idx, charge in charge_dict.items():
        if atom_idx < len(atom_data):
            atom_data[atom_idx]["charge"] = float(charge)

    # Build atoms
    atoms = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
            charge=a.get("charge", 0.0),
        )
        for a in atom_data
    ]

    # Build bonds
    bonds = [
        {
            "atom1": b[0],
            "atom2": b[1],
            "order": b[2],
        }
        for b in bond_data
    ]

    # Build metadata
    metadata = {
        "source_format": "sdf",
        "source_file": source_file,
        "molecule_name": name,
        "molecule_index": mol_index,
        "format_version": "V3000" if is_v3000 else "V2000",
    }

    return (ChemicalSystem(atoms=atoms, bonds=bonds, metadata=metadata), end_idx)
