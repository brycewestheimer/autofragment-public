# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
VASP POSCAR/CONTCAR file reader.

This module provides functions for reading VASP structure files (POSCAR/CONTCAR)
into ChemicalSystem objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem


def read_poscar(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read VASP POSCAR file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to POSCAR file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.

    Notes
    -----
    POSCAR format:
    - Line 1: Comment
    - Line 2: Universal scaling factor
    - Lines 3-5: Lattice vectors
    - Line 6: Element symbols (VASP 5+) or counts
    - Line 7: Numbers of atoms per element type
    - Line 8: Selective dynamics (optional) or coordinate type
    - Line 9+: Coordinates
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"POSCAR file not found: {filepath}")

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 8:
        raise ValueError(f"POSCAR file too short: {filepath}")

    # Line 1: Comment
    comment = lines[0].strip()

    # Line 2: Scaling factor
    scaling_factor = float(lines[1].strip())

    # Lines 3-5: Lattice vectors
    lattice = np.zeros((3, 3))
    for i in range(3):
        parts = lines[2 + i].split()
        lattice[i] = [float(x) for x in parts[:3]]
    lattice *= scaling_factor

    # Line 6: Element symbols or atom counts
    line6 = lines[5].split()

    # Check if line 6 is element symbols or counts
    try:
        counts = [int(x) for x in line6]
        # Line 6 was counts (old VASP 4 format)
        elements: List[str] = []
        # Try to extract from comment
        comment_parts = comment.split()
        if len(comment_parts) >= len(counts):
            elements = comment_parts[:len(counts)]
        else:
            elements = [f"X{i+1}" for i in range(len(counts))]
        counts_line = 5
    except ValueError:
        # Line 6 is element symbols (VASP 5+ format)
        elements = line6
        # Line 7 is counts
        counts = [int(x) for x in lines[6].split()]
        counts_line = 6

    # Expand elements based on counts
    expanded_elements: List[str] = []
    for elem, count in zip(elements, counts):
        expanded_elements.extend([elem] * count)

    total_atoms = sum(counts)

    # Find coordinate type line
    coord_line_idx = counts_line + 1
    coord_type_line = lines[coord_line_idx].strip().lower()

    # Check for selective dynamics
    selective_dynamics = False
    if coord_type_line.startswith("s"):
        selective_dynamics = True
        coord_line_idx += 1
        coord_type_line = lines[coord_line_idx].strip().lower()

    # Determine coordinate type (Direct/Cartesian)
    is_cartesian = coord_type_line.startswith(("c", "k"))

    # Parse coordinates
    coords_start = coord_line_idx + 1
    coords: List[np.ndarray] = []

    for i in range(total_atoms):
        if coords_start + i >= len(lines):
            break
        parts = lines[coords_start + i].split()
        if len(parts) < 3:
            break

        pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])])

        if not is_cartesian:
            # Convert fractional to Cartesian
            pos = pos @ lattice
        else:
            pos *= scaling_factor

        coords.append(pos)

    # Build atoms
    atoms = [
        Atom(
            symbol=elem,
            coords=coord,
        )
        for elem, coord in zip(expanded_elements, coords)
    ]

    # Build metadata
    metadata: Dict[str, Any] = {
        "source_format": "poscar",
        "source_file": str(path.name),
        "comment": comment,
        "lattice": lattice.tolist(),
        "selective_dynamics": selective_dynamics,
        "coordinate_type": "cartesian" if is_cartesian else "direct",
    }

    return ChemicalSystem(atoms=atoms, bonds=[], metadata=metadata)


def read_contcar(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read VASP CONTCAR file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to CONTCAR file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.

    Notes
    -----
    CONTCAR has the same format as POSCAR.
    """
    return read_poscar(filepath)
