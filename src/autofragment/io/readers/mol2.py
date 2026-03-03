# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tripos MOL2 file reader.

This module provides functions for reading Tripos MOL2 files into ChemicalSystem objects,
including support for Tripos atom types, bond orders, and partial charges.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem

# Mapping of Tripos atom types to element symbols
# Based on Tripos force field atom type definitions
_TRIPOS_TYPE_TO_ELEMENT = {
    # Carbon types
    "C.3": "C", "C.2": "C", "C.1": "C", "C.ar": "C", "C.cat": "C",
    # Nitrogen types
    "N.3": "N", "N.2": "N", "N.1": "N", "N.ar": "N", "N.am": "N",
    "N.pl3": "N", "N.4": "N",
    # Oxygen types
    "O.3": "O", "O.2": "O", "O.co2": "O", "O.spc": "O", "O.t3p": "O",
    # Sulfur types
    "S.3": "S", "S.2": "S", "S.O": "S", "S.O2": "S", "S.ar": "S",
    # Phosphorus types
    "P.3": "P",
    # Hydrogen types
    "H": "H", "H.spc": "H", "H.t3p": "H",
    # Halogens
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    # Other common elements
    "Li": "Li", "Na": "Na", "K": "K",
    "Mg": "Mg", "Ca": "Ca",
    "Fe": "Fe", "Zn": "Zn", "Cu": "Cu",
    "Si": "Si", "Se": "Se",
    # Lone pair and dummy atoms
    "LP": "LP", "Du": "Du",
    "ANY": "X", "HEV": "X", "HET": "X",
}

# Bond type mapping (MOL2 to order)
_BOND_TYPE_TO_ORDER = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "ar": 1.5,  # Aromatic
    "am": 1.0,  # Amide
    "du": 0.0,  # Dummy
    "un": 1.0,  # Unknown
    "nc": 0.0,  # Not connected
}


def _element_from_tripos_type(tripos_type: str) -> str:
    """
    Extract element symbol from Tripos atom type.

    Parameters
    ----------
    tripos_type : str
        Tripos atom type (e.g., "C.3", "N.ar", "H").

    Returns
    -------
    str
        Element symbol.
    """
    # Try direct lookup first
    if tripos_type in _TRIPOS_TYPE_TO_ELEMENT:
        return _TRIPOS_TYPE_TO_ELEMENT[tripos_type]

    # Try uppercase
    if tripos_type.upper() in _TRIPOS_TYPE_TO_ELEMENT:
        return _TRIPOS_TYPE_TO_ELEMENT[tripos_type.upper()]

    # Extract base element (before the dot)
    base = tripos_type.split(".")[0].strip()

    if base in _TRIPOS_TYPE_TO_ELEMENT:
        return _TRIPOS_TYPE_TO_ELEMENT[base]

    # Capitalize and check if it's a valid element
    if len(base) == 1:
        return base.upper()
    elif len(base) == 2:
        return base[0].upper() + base[1].lower()
    elif len(base) > 2:
        # Take first two characters
        return base[0].upper() + base[1].lower()

    return "X"  # Unknown element


def _parse_mol2_atom(line: str) -> Dict[str, Any]:
    """
    Parse MOL2 ATOM section line.

    Format: atom_id atom_name x y z atom_type [subst_id subst_name charge [status_bit]]

    Parameters
    ----------
    line : str
        A line from the @<TRIPOS>ATOM section.

    Returns
    -------
    dict
        Parsed atom data.
    """
    parts = line.split()

    if len(parts) < 6:
        raise ValueError(f"Invalid MOL2 atom line (need at least 6 fields): {line!r}")

    atom_id = int(parts[0])
    atom_name = parts[1]
    x = float(parts[2])
    y = float(parts[3])
    z = float(parts[4])
    atom_type = parts[5]

    # Optional fields
    subst_id = int(parts[6]) if len(parts) > 6 else 1
    subst_name = parts[7] if len(parts) > 7 else ""
    charge = float(parts[8]) if len(parts) > 8 else 0.0
    status_bit = parts[9] if len(parts) > 9 else ""

    element = _element_from_tripos_type(atom_type)

    return {
        "id": atom_id,
        "name": atom_name,
        "x": x,
        "y": y,
        "z": z,
        "atom_type": atom_type,
        "element": element,
        "subst_id": subst_id,
        "subst_name": subst_name,
        "charge": charge,
        "status_bit": status_bit,
    }


def _parse_mol2_bond(line: str) -> Tuple[int, int, float]:
    """
    Parse MOL2 BOND section line.

    Format: bond_id origin_atom_id target_atom_id bond_type [status_bits]
    Bond types: 1, 2, 3, ar, am, du, nc, un

    Parameters
    ----------
    line : str
        A line from the @<TRIPOS>BOND section.

    Returns
    -------
    tuple
        (atom1_index, atom2_index, bond_order) with 0-based indices.
    """
    parts = line.split()

    if len(parts) < 4:
        raise ValueError(f"Invalid MOL2 bond line (need at least 4 fields): {line!r}")

    # bond_id = int(parts[0])  # Not used
    origin = int(parts[1])
    target = int(parts[2])
    bond_type = parts[3].lower()

    order = _BOND_TYPE_TO_ORDER.get(bond_type, 1.0)

    # Convert to 0-based indices
    return (origin - 1, target - 1, order)


def read_mol2(
    filepath: Union[str, Path],
    molecule_index: int = 0,
) -> ChemicalSystem:
    """
    Read Tripos MOL2 file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to MOL2 file.
    molecule_index : int, optional
        Which molecule to read for multi-molecule files (0-indexed).
        Default is 0.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms and bonds.

    Examples
    --------
    >>> system = read_mol2("molecule.mol2")
    >>> print(f"Read {system.n_atoms} atoms with {system.n_bonds} bonds")
    """
    molecules = list(read_mol2_multi(filepath))

    if not molecules:
        raise ValueError(f"No molecules found in MOL2 file: {filepath}")

    if molecule_index >= len(molecules):
        raise IndexError(
            f"Molecule index {molecule_index} out of range "
            f"(file contains {len(molecules)} molecules)"
        )

    return molecules[molecule_index]


def read_mol2_multi(filepath: Union[str, Path]) -> List[ChemicalSystem]:
    """
    Read all molecules from a MOL2 file.

    Parameters
    ----------
    filepath : str or Path
        Path to MOL2 file.

    Yields
    ------
    ChemicalSystem
        ChemicalSystem for each molecule in the file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"MOL2 file not found: {filepath}")

    molecules: List[ChemicalSystem] = []
    current_molecule: Optional[Dict[str, Any]] = None
    current_section: Optional[str] = None

    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line_stripped = line.strip()

            # Check for section headers
            if line_stripped.startswith("@<TRIPOS>"):
                section_name = line_stripped[9:].upper()
                current_section = section_name

                if section_name == "MOLECULE":
                    # Start new molecule (save previous if exists)
                    if current_molecule is not None:
                        mol = _build_chemical_system(current_molecule, str(path.name))
                        molecules.append(mol)
                    current_molecule = {
                        "name": "",
                        "atoms": [],
                        "bonds": [],
                        "n_atoms": 0,
                        "n_bonds": 0,
                        "mol_type": "",
                        "charge_type": "",
                    }
                continue

            if current_molecule is None:
                continue

            if not line_stripped:
                continue

            # Parse based on current section
            if current_section == "MOLECULE":
                # First line is molecule name
                if not current_molecule["name"]:
                    current_molecule["name"] = line_stripped
                elif current_molecule["n_atoms"] == 0:
                    # Second line has counts
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        try:
                            current_molecule["n_atoms"] = int(parts[0])
                            current_molecule["n_bonds"] = int(parts[1])
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid MOL2 molecule counts line at {path.name}:{line_no}: {line_stripped!r}"
                            ) from e
                    else:
                        warnings.warn(
                            (
                                f"Skipping incomplete MOL2 molecule counts line at "
                                f"{path.name}:{line_no}: {line_stripped!r}"
                            ),
                            UserWarning,
                            stacklevel=2,
                        )
                elif not current_molecule["mol_type"]:
                    current_molecule["mol_type"] = line_stripped
                elif not current_molecule["charge_type"]:
                    current_molecule["charge_type"] = line_stripped

            elif current_section == "ATOM":
                try:
                    atom_data = _parse_mol2_atom(line_stripped)
                    current_molecule["atoms"].append(atom_data)
                except ValueError as e:
                    raise ValueError(
                        f"Malformed MOL2 atom record at {path.name}:{line_no}: {line_stripped!r}"
                    ) from e

            elif current_section == "BOND":
                try:
                    bond_data = _parse_mol2_bond(line_stripped)
                    current_molecule["bonds"].append(bond_data)
                except ValueError as e:
                    raise ValueError(
                        f"Malformed MOL2 bond record at {path.name}:{line_no}: {line_stripped!r}"
                    ) from e

    # Don't forget the last molecule
    if current_molecule is not None:
        mol = _build_chemical_system(current_molecule, str(path.name))
        molecules.append(mol)

    return molecules


def _build_chemical_system(mol_data: Dict[str, Any], source_file: str) -> ChemicalSystem:
    """
    Build a ChemicalSystem from parsed MOL2 data.

    Parameters
    ----------
    mol_data : dict
        Parsed molecule data with atoms and bonds.
    source_file : str
        Name of source file for metadata.

    Returns
    -------
    ChemicalSystem
        Complete chemical system.
    """
    # Build atoms list
    atoms = [
        Atom(
            symbol=rec["element"],
            coords=np.array([rec["x"], rec["y"], rec["z"]]),
            charge=rec["charge"],
        )
        for rec in mol_data["atoms"]
    ]

    # Build bonds list
    bonds = [
        {
            "atom1": b[0],
            "atom2": b[1],
            "order": b[2],
        }
        for b in mol_data["bonds"]
    ]

    # Build metadata
    metadata = {
        "source_format": "mol2",
        "source_file": source_file,
        "molecule_name": mol_data["name"],
        "mol_type": mol_data["mol_type"],
        "charge_type": mol_data["charge_type"],
    }

    # Add atom type info
    if mol_data["atoms"]:
        atom_info = [
            {
                "atom_name": rec["name"],
                "atom_type": rec["atom_type"],
                "subst_id": rec["subst_id"],
                "subst_name": rec["subst_name"],
            }
            for rec in mol_data["atoms"]
        ]
        metadata["atom_info"] = atom_info

    return ChemicalSystem(atoms=atoms, bonds=bonds, metadata=metadata)
