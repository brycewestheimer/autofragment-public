# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""XYZ file parsing for water clusters.

This module provides functions for reading XYZ files containing
water molecules (O-H-H triplets).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import numpy as np

from autofragment.core.types import (
    Atom,
    ChemicalSystem,
    Molecule,
    molecules_to_system,
    system_to_molecules,
)

# Conversion factor from Bohr to Angstrom
_BOHR_TO_ANGSTROM = 0.529177210903


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def _require(condition: bool, message: str) -> None:
    """Assert a condition or raise ValidationError."""
    if not condition:
        raise ValidationError(message)


def read_xyz(
    filepath: str | Path,
    *,
    xyz_units: Literal["angstrom", "bohr"] = "angstrom",
    atoms_per_molecule: int | None = 3,
    validate_water: bool = True,
) -> ChemicalSystem:
    """
    Parse an XYZ file into a ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to the XYZ file.
    xyz_units : {"angstrom", "bohr"}, optional
        Units of the coordinates. Default is "angstrom".
    atoms_per_molecule : int or None, optional
        Number of atoms per molecule. Default is 3 (for water).
        If None, treats all atoms as a single molecule.
    validate_water : bool, optional
        If True, validate that each molecule is O-H-H. Default is True.
        Ignored if atoms_per_molecule is None.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with molecule boundaries stored in metadata.
    """
    path = Path(filepath)
    _require(path.exists(), f"XYZ file not found: {filepath}")

    text = path.read_text().splitlines()

    # Find first non-empty line (atom count)
    start_idx = 0
    while start_idx < len(text) and not text[start_idx].strip():
        start_idx += 1

    _require(start_idx < len(text), f"XYZ file not found or empty: {filepath}")

    lines = []
    # 1. Count line
    lines.append(text[start_idx].strip())

    # 2. Comment line (if exists)
    if start_idx + 1 < len(text):
        lines.append(text[start_idx + 1].strip())

        # 3. Atom lines (skip empty ones)
        for ln in text[start_idx + 2:]:
            s = ln.strip()
            if s:
                lines.append(s)

    _require(len(lines) >= 2, f"XYZ file too short: {filepath}")

    try:
        atom_count = int(lines[0])
    except Exception as e:
        raise ValidationError(f"First line must be atom count in {filepath}: {e}")

    atom_lines = lines[2 : 2 + atom_count]
    _require(
        len(atom_lines) == atom_count,
        f"Atom count mismatch in {filepath}: expected {atom_count}, got {len(atom_lines)}",
    )

    if atoms_per_molecule is None:
        chunk_size = atom_count
        validate_water = False  # Disable water validation for generic single molecule
    else:
        _require(
            atoms_per_molecule > 0,
            f"atoms_per_molecule must be positive, got {atoms_per_molecule}",
        )
        chunk_size = atoms_per_molecule
        _require(
            atom_count % chunk_size == 0,
            f"Atom count must be multiple of {chunk_size} in {filepath}: {atom_count}",
        )

    scale = 1.0 if xyz_units == "angstrom" else _BOHR_TO_ANGSTROM

    molecules: List[Molecule] = []

    for i in range(0, len(atom_lines), chunk_size):
        mol_lines = atom_lines[i : i + chunk_size]
        atoms: Molecule = []

        for j, ln in enumerate(mol_lines):
            parts = ln.split()
            _require(
                len(parts) >= 4,
                f"Bad atom line in {filepath}: {ln!r}",
            )
            symbol = parts[0]
            coords = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            ) * scale
            atoms.append(Atom(symbol=symbol, coords=coords))

        # Validate water structure
        if validate_water and atoms_per_molecule == 3:
            mol_idx = i // chunk_size
            _require(
                atoms[0].symbol.upper() == "O",
                f"Expected O as first atom in molecule {mol_idx} at atoms {i}-{i+2} in {filepath}",
            )
            _require(
                atoms[1].symbol.upper() == "H",
                f"Expected H as second atom in molecule {mol_idx} at atoms {i}-{i+2} in {filepath}",
            )
            _require(
                atoms[2].symbol.upper() == "H",
                f"Expected H as third atom in molecule {mol_idx} at atoms {i}-{i+2} in {filepath}",
            )

        molecules.append(atoms)

    metadata: dict[str, object] = {
        "source_file": str(path),
        "xyz_units": xyz_units,
    }
    if atoms_per_molecule is not None:
        metadata["atoms_per_molecule"] = atoms_per_molecule

    return molecules_to_system(molecules, metadata=metadata)


def read_xyz_molecules(
    filepath: str | Path,
    *,
    xyz_units: Literal["angstrom", "bohr"] = "angstrom",
    atoms_per_molecule: int | None = 3,
    validate_water: bool = True,
) -> List[Molecule]:
    """Parse an XYZ file into isolated molecules (explicit boundary helper)."""
    system = read_xyz(
        filepath,
        xyz_units=xyz_units,
        atoms_per_molecule=atoms_per_molecule,
        validate_water=validate_water,
    )
    return system_to_molecules(system, require_metadata=True)


def write_xyz(
    filepath: str | Path,
    system: ChemicalSystem | List[Molecule],
    comment: str = "",
    *,
    atoms_per_molecule: int | None = None,
) -> None:
    """
    Write a system to an XYZ file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    system : ChemicalSystem | List[Molecule]
        System to write. Molecule lists are treated as isolated fragments.
    comment : str, optional
        Comment line (line 2 of XYZ file). Default is empty.
    atoms_per_molecule : int, optional
        When writing a ChemicalSystem without molecule metadata, chunk atoms
        into fixed-size molecules.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(system, ChemicalSystem):
        molecules = system_to_molecules(
            system,
            atoms_per_molecule=atoms_per_molecule,
            require_metadata=False,
        )
    else:
        molecules = system

    lines = []
    atom_count = sum(len(mol) for mol in molecules)
    lines.append(str(atom_count))
    lines.append(comment)

    for mol in molecules:
        for atom in mol:
            x, y, z = atom.coords
            lines.append(f"{atom.symbol:2s} {x:15.10f} {y:15.10f} {z:15.10f}")

    path.write_text("\n".join(lines) + "\n")
