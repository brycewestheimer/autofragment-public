# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
QCSchema JSON file reader.

This module provides functions for reading QCSchema JSON files (MolSSI standard format)
into ChemicalSystem objects.

Reference: https://molssi-qc-schema.readthedocs.io/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from autofragment.core.types import Atom, ChemicalSystem

# Conversion factor from Bohr to Angstrom
_BOHR_TO_ANGSTROM = 0.529177210903


def qcschema_to_system(data: Dict[str, Any]) -> ChemicalSystem:
    """
    Convert QCSchema dictionary to ChemicalSystem.

    Handles both QCSchema molecule and QCElemental Molecule formats.

    Parameters
    ----------
    data : dict
        QCSchema-format dictionary with keys like 'symbols', 'geometry', etc.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms and optional bonds.

    Notes
    -----
    QCSchema geometry is in Bohr; this function converts to Angstroms.
    """
    # Validate schema
    schema_name = data.get("schema_name", "")
    if schema_name and schema_name not in ("qcschema_molecule", "qc_schema_molecule"):
        # Check if it might be a result or input schema
        if "molecule" in data:
            return qcschema_to_system(data["molecule"])

    # Get symbols (required)
    symbols = data.get("symbols", [])
    if not symbols:
        raise ValueError("QCSchema data missing 'symbols' field")

    # Get geometry (required, in Bohr as flat list)
    geometry = data.get("geometry", [])
    if not geometry:
        raise ValueError("QCSchema data missing 'geometry' field")

    # Convert geometry to Angstroms
    coords = np.array(geometry, dtype=float).reshape(-1, 3) * _BOHR_TO_ANGSTROM

    if len(coords) != len(symbols):
        raise ValueError(
            f"Mismatch between symbols ({len(symbols)}) and "
            f"geometry ({len(coords)}) lengths"
        )

    # Get optional fields
    molecular_charge = data.get("molecular_charge", 0)
    molecular_multiplicity = data.get("molecular_multiplicity", 1)

    # Get atomic masses and numbers if present
    masses = data.get("masses", [])
    atomic_numbers = data.get("atomic_numbers", [])

    # Get connectivity (bonds)
    connectivity = data.get("connectivity", [])

    # Get fragments if present
    fragments = data.get("fragments", [])
    fragment_charges = data.get("fragment_charges", [])
    fragment_multiplicities = data.get("fragment_multiplicities", [])

    # Get real/ghost information
    real = data.get("real", [True] * len(symbols))

    # Build atoms
    atoms = [
        Atom(
            symbol=sym,
            coords=coords[i],
            charge=0.0,  # QCSchema doesn't store partial charges
        )
        for i, sym in enumerate(symbols)
    ]

    # Build bonds from connectivity
    bonds: List[Dict[str, Any]] = []
    for bond_info in connectivity:
        if len(bond_info) >= 3:
            atom1, atom2, order = bond_info[0], bond_info[1], bond_info[2]
            bonds.append({
                "atom1": int(atom1),
                "atom2": int(atom2),
                "order": float(order),
            })

    # Build metadata
    metadata: Dict[str, Any] = {
        "source_format": "qcschema",
        "molecular_charge": molecular_charge,
        "molecular_multiplicity": molecular_multiplicity,
        "schema_version": data.get("schema_version", 2),
    }

    if masses:
        metadata["masses"] = masses
    if atomic_numbers:
        metadata["atomic_numbers"] = atomic_numbers
    if fragments:
        metadata["fragments"] = fragments
        metadata["fragment_charges"] = fragment_charges
        metadata["fragment_multiplicities"] = fragment_multiplicities
    if any(not r for r in real):
        metadata["real"] = real

    # Copy any additional fields
    for key in ["name", "identifiers", "comment", "provenance"]:
        if key in data:
            metadata[key] = data[key]

    return ChemicalSystem(atoms=atoms, bonds=bonds, metadata=metadata)


def read_qcschema(filepath: Union[str, Path]) -> ChemicalSystem:
    """
    Read QCSchema JSON file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to QCSchema JSON file.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms and optional bonds.

    Examples
    --------
    >>> system = read_qcschema("molecule.json")
    >>> print(f"Read {system.n_atoms} atoms, charge = {system.metadata['molecular_charge']}")

    Notes
    -----
    This function handles:
    - qcschema_molecule (standard molecule schema)
    - QCElemental Molecule objects serialized to JSON
    - QCSchema results that contain a molecule field
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"QCSchema file not found: {filepath}")

    with open(path, "r") as f:
        data = json.load(f)

    # Add source file to result
    result = qcschema_to_system(data)
    result.metadata["source_file"] = str(path.name)

    return result


def read_qcschema_multi(filepath: Union[str, Path]) -> List[ChemicalSystem]:
    """
    Read multiple molecules from a QCSchema JSON array.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file containing an array of QCSchema molecules.

    Returns
    -------
    list
        List of ChemicalSystem objects.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"QCSchema file not found: {filepath}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [qcschema_to_system(mol) for mol in data]
    else:
        return [qcschema_to_system(data)]
