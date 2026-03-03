# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantum chemistry program input file parsers.

This module provides readers for various QC program input formats:
- GAMESS (.inp)
- Psi4 (.dat, .py)
- Q-Chem (.in)
- ORCA (.inp)
- NWChem (.nw)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem

# Element symbol to atomic number mapping
_ELEMENT_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}

# Atomic number to element symbol
_Z_TO_ELEMENT = {v: k for k, v in _ELEMENT_TO_Z.items()}


# ============================================================================
#                              GAMESS Reader
# ============================================================================

def _parse_gamess_groups(filepath: Path) -> Dict[str, List[str]]:
    """
    Parse GAMESS $GROUP ... $END blocks.

    Parameters
    ----------
    filepath : Path
        Path to GAMESS input file.

    Returns
    -------
    dict
        Dictionary mapping group name (e.g., "$CONTRL") to list of lines.
    """
    groups: Dict[str, List[str]] = {}
    current_group: Optional[str] = None
    current_lines: List[str] = []

    with open(filepath, "r") as f:
        for line in f:
            # Skip comment lines (! at start)
            stripped = line.strip()
            if stripped.startswith("!"):
                continue

            # Look for $GROUP markers
            upper = stripped.upper()

            # Check for $GROUP start (but not $END)
            if "$" in upper and not upper.startswith("$END"):
                # Find the group name
                match = re.search(r'\$(\w+)', upper)
                if match:
                    group_name = f"${match.group(1)}"

                    # Save previous group if any
                    if current_group is not None and current_lines:
                        groups[current_group] = current_lines

                    current_group = group_name
                    current_lines = [line.rstrip()]

            elif "$END" in upper:
                # End of current group
                if current_group is not None:
                    current_lines.append(line.rstrip())
                    groups[current_group] = current_lines
                    current_group = None
                    current_lines = []

            elif current_group is not None:
                # Inside a group
                current_lines.append(line.rstrip())

    return groups


def _parse_gamess_namelist(lines: List[str]) -> Dict[str, Any]:
    """
    Parse GAMESS namelist-style group into key-value pairs.

    Parameters
    ----------
    lines : list
        Lines of a single GAMESS group.

    Returns
    -------
    dict
        Parsed key-value pairs (lowercase keys).
    """
    result: Dict[str, Any] = {}

    # Concatenate all lines and remove $GROUP and $END
    text = " ".join(lines)

    # Remove $GROUP_NAME and $END
    text = re.sub(r'\$\w+', '', text)
    text = re.sub(r'\$END', '', text, flags=re.IGNORECASE)

    # Parse key=value pairs
    # Handle arrays like INDAT(1)=...
    pattern = r'(\w+(?:\([^)]+\))?)\s*=\s*([^=]+?)(?=\s+\w+(?:\([^)]+\))?\s*=|\s*$)'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        key = match.group(1).lower()
        value = match.group(2).strip().rstrip(",")

        # Try to parse as number
        try:
            if "." in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
        except ValueError:
            # Keep as string
            result[key] = value

    return result


def _parse_gamess_data(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse GAMESS $DATA group for geometry.

    Format:
    $DATA
    Title
    Symmetry (C1 for no symmetry)
    Element  Znuc  X  Y  Z
    ...
    $END

    Parameters
    ----------
    lines : list
        Lines of the $DATA group.

    Returns
    -------
    list
        List of atom dictionaries.
    """
    atoms = []

    # Skip header lines ($DATA, title, symmetry)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("$DATA"):
            start_idx = i + 1
            break

    # Skip title and symmetry lines
    if start_idx < len(lines):
        start_idx += 2  # title + symmetry

    # Parse atom lines
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()

        if not line or line.upper().startswith("$END"):
            break

        parts = line.split()
        if len(parts) < 5:
            continue

        # Format: ELEMENT ZNUC X Y Z [BASIS_INFO...]
        try:
            element = parts[0]
            znuc = float(parts[1])  # Atomic number (can be float)
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            # Get element from znuc if element is generic
            if element.upper() in ("BQ", "X") or not element.isalpha():
                element = _Z_TO_ELEMENT.get(int(znuc), "X")

            atoms.append({
                "element": element,
                "x": x,
                "y": y,
                "z": z,
                "znuc": znuc,
            })
        except (ValueError, IndexError) as e:
            raise ValueError(f"Malformed GAMESS atom record: {line!r}") from e

    return atoms


def read_gamess_input(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read GAMESS input file (.inp).

    Parameters
    ----------
    filepath : str or Path
        Path to GAMESS input file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms from $DATA group.

    Examples
    --------
    >>> system = read_gamess_input("water.inp")
    >>> print(f"Read {system.n_atoms} atoms")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GAMESS input file not found: {filepath}")

    groups = _parse_gamess_groups(path)

    # Parse $DATA for geometry
    if "$DATA" not in groups:
        raise ValueError(f"No $DATA group found in GAMESS file: {filepath}")

    atom_data = _parse_gamess_data(groups["$DATA"])

    # Build atoms
    atoms = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
        )
        for a in atom_data
    ]

    # Parse $CONTRL for charge/multiplicity
    charge = 0
    multiplicity = 1
    if "$CONTRL" in groups:
        contrl = _parse_gamess_namelist(groups["$CONTRL"])
        charge = contrl.get("icharg", 0)
        multiplicity = contrl.get("mult", 1)

    # Build metadata
    metadata: Dict[str, Any] = {
        "source_format": "gamess",
        "source_file": str(path.name),
        "molecular_charge": charge,
        "molecular_multiplicity": multiplicity,
    }

    # Add FMO info if present
    if "$FMO" in groups:
        fmo = _parse_gamess_namelist(groups["$FMO"])
        metadata["fmo"] = fmo

    return ChemicalSystem(atoms=atoms, bonds=[], metadata=metadata)


# ============================================================================
#                              Psi4 Reader
# ============================================================================

def _parse_psi4_molecule(text: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Parse Psi4 molecule block.

    Parameters
    ----------
    text : str
        Content between molecule { and }.

    Returns
    -------
    tuple
        (atoms, charge, multiplicity)
    """
    atoms: List[Dict[str, Any]] = []
    charge = 0
    multiplicity = 1

    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Fragment separator
        if line == "--":
            # Next charge/mult applies to new fragment
            continue

        # Check for charge/multiplicity line (two integers)
        parts = line.split()
        if len(parts) == 2:
            try:
                c = int(parts[0])
                m = int(parts[1])
                # This is a charge/multiplicity line
                charge += c  # Accumulate for total
                multiplicity = m  # Use last multiplicity
                continue
            except ValueError:
                pass

        # Parse atom line: Element X Y Z [units]
        if len(parts) >= 4:
            try:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                atoms.append({
                    "element": element,
                    "x": x,
                    "y": y,
                    "z": z,
                })
            except ValueError as e:
                raise ValueError(f"Malformed Psi4 coordinate line: {line!r}") from e

    return atoms, charge, multiplicity


def read_psi4_input(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read Psi4 input file.

    Parameters
    ----------
    filepath : str or Path
        Path to Psi4 input file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.

    Examples
    --------
    >>> system = read_psi4_input("water.dat")
    >>> print(f"Read {system.n_atoms} atoms")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Psi4 input file not found: {filepath}")

    content = path.read_text()

    # Find molecule block
    mol_pattern = r'molecule\s+(\w+)?\s*\{([^}]+)\}'
    match = re.search(mol_pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"No molecule block found in Psi4 file: {filepath}")

    mol_name = match.group(1) or "molecule"
    mol_content = match.group(2)

    atom_data, charge, multiplicity = _parse_psi4_molecule(mol_content)

    # Build atoms
    atoms = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
        )
        for a in atom_data
    ]

    # Build metadata
    metadata: Dict[str, Any] = {
        "source_format": "psi4",
        "source_file": str(path.name),
        "molecule_name": mol_name,
        "molecular_charge": charge,
        "molecular_multiplicity": multiplicity,
    }

    return ChemicalSystem(atoms=atoms, bonds=[], metadata=metadata)


# ============================================================================
#                              Q-Chem Reader
# ============================================================================

def read_qchem_input(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read Q-Chem input file.

    Parameters
    ----------
    filepath : str or Path
        Path to Q-Chem input file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Q-Chem input file not found: {filepath}")

    content = path.read_text()

    # Find $molecule section
    mol_pattern = r'\$molecule\s*\n(.*?)\$end'
    match = re.search(mol_pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"No $molecule section found in Q-Chem file: {filepath}")

    mol_content = match.group(1).strip()
    lines = mol_content.split("\n")

    # First line is charge and multiplicity
    charge = 0
    multiplicity = 1
    atoms: List[Dict[str, Any]] = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        parts = line.split()

        if i == 0 and len(parts) == 2:
            # Charge and multiplicity
            try:
                charge = int(parts[0])
                multiplicity = int(parts[1])
                continue
            except ValueError as e:
                raise ValueError(
                    f"Invalid Q-Chem charge/multiplicity line: {line!r}"
                ) from e

        # Atom line
        if len(parts) >= 4:
            try:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                atoms.append({
                    "element": element,
                    "x": x,
                    "y": y,
                    "z": z,
                })
            except ValueError as e:
                raise ValueError(f"Malformed Q-Chem coordinate line: {line!r}") from e
        elif i > 0:
            warnings.warn(
                f"Skipping non-coordinate line in Q-Chem $molecule block: {line!r}",
                UserWarning,
                stacklevel=2,
            )

    # Build atoms
    atom_list = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
        )
        for a in atoms
    ]

    metadata: Dict[str, Any] = {
        "source_format": "qchem",
        "source_file": str(path.name),
        "molecular_charge": charge,
        "molecular_multiplicity": multiplicity,
    }

    return ChemicalSystem(atoms=atom_list, bonds=[], metadata=metadata)


# ============================================================================
#                              ORCA Reader
# ============================================================================

def read_orca_input(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read ORCA input file.

    Parameters
    ----------
    filepath : str or Path
        Path to ORCA input file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ORCA input file not found: {filepath}")

    content = path.read_text()

    # Find *xyz or * xyzfile block
    xyz_pattern = r'\*\s*xyz\s+(-?\d+)\s+(\d+)\s*\n(.*?)\*'
    match = re.search(xyz_pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"No *xyz block found in ORCA file: {filepath}")

    charge = int(match.group(1))
    multiplicity = int(match.group(2))
    coord_content = match.group(3).strip()

    atoms: List[Dict[str, Any]] = []

    for line in coord_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                atoms.append({
                    "element": element,
                    "x": x,
                    "y": y,
                    "z": z,
                })
            except ValueError as e:
                raise ValueError(f"Malformed ORCA coordinate line: {line!r}") from e

    # Build atoms
    atom_list = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
        )
        for a in atoms
    ]

    metadata: Dict[str, Any] = {
        "source_format": "orca",
        "source_file": str(path.name),
        "molecular_charge": charge,
        "molecular_multiplicity": multiplicity,
    }

    return ChemicalSystem(atoms=atom_list, bonds=[], metadata=metadata)


# ============================================================================
#                              NWChem Reader
# ============================================================================

def read_nwchem_input(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read NWChem input file.

    Parameters
    ----------
    filepath : str or Path
        Path to NWChem input file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"NWChem input file not found: {filepath}")

    content = path.read_text()

    # Find geometry block
    geom_pattern = r'geometry.*?\n(.*?)(end|\Z)'
    match = re.search(geom_pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        raise ValueError(f"No geometry block found in NWChem file: {filepath}")

    coord_content = match.group(1).strip()

    atoms: List[Dict[str, Any]] = []

    for line in coord_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Skip keywords
        if any(kw in line.lower() for kw in ["units", "noautoz", "print", "symmetry"]):
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                atoms.append({
                    "element": element,
                    "x": x,
                    "y": y,
                    "z": z,
                })
            except ValueError as e:
                raise ValueError(f"Malformed NWChem coordinate line: {line!r}") from e

    # Find charge
    charge = 0
    multiplicity = 1

    charge_match = re.search(r'charge\s+(-?\d+)', content, re.IGNORECASE)
    if charge_match:
        charge = int(charge_match.group(1))

    # Build atoms
    atom_list = [
        Atom(
            symbol=a["element"],
            coords=np.array([a["x"], a["y"], a["z"]]),
        )
        for a in atoms
    ]

    metadata: Dict[str, Any] = {
        "source_format": "nwchem",
        "source_file": str(path.name),
        "molecular_charge": charge,
        "molecular_multiplicity": multiplicity,
    }

    return ChemicalSystem(atoms=atom_list, bonds=[], metadata=metadata)
