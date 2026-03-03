# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
QCSchema JSON writer with fragment annotations.

This module provides functions for writing ChemicalSystem and Fragment data
to QCSchema-compliant JSON format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from autofragment.core.types import ChemicalSystem, Fragment

# Conversion factor from Angstrom to Bohr
_ANGSTROM_TO_BOHR = 1.8897259886


def system_to_qcschema(
    system: ChemicalSystem,
    fragments: Optional[List[Fragment]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert ChemicalSystem to QCSchema dictionary.

    Parameters
    ----------
    system : ChemicalSystem
        The chemical system to convert.
    fragments : list, optional
        Fragment definitions to include.
    name : str, optional
        Name for the molecule.

    Returns
    -------
    dict
        QCSchema-compliant molecule dictionary.

    Notes
    -----
    QCSchema requires geometry in Bohr; this function converts from Angstroms.
    """
    # Extract symbols
    symbols = [atom.symbol for atom in system.atoms]

    # Convert coordinates to Bohr
    coords_angstrom = np.array([atom.coords for atom in system.atoms])
    coords_bohr = coords_angstrom * _ANGSTROM_TO_BOHR
    geometry = coords_bohr.flatten().tolist()

    # Build connectivity from bonds
    connectivity = []
    for bond in system.bonds:
        a1 = bond["atom1"]
        a2 = bond["atom2"]
        order = bond.get("order", 1.0)
        connectivity.append([int(a1), int(a2), float(order)])

    # Get charge and multiplicity from metadata or default
    molecular_charge = system.metadata.get("molecular_charge", 0)
    molecular_multiplicity = system.metadata.get("molecular_multiplicity", 1)

    result: Dict[str, Any] = {
        "schema_name": "qcschema_molecule",
        "schema_version": 2,
        "symbols": symbols,
        "geometry": geometry,
        "molecular_charge": molecular_charge,
        "molecular_multiplicity": molecular_multiplicity,
    }

    if connectivity:
        result["connectivity"] = connectivity

    if name:
        result["name"] = name

    # Add fragment information if provided
    if fragments:
        # Build fragment atom indices
        frag_indices = []
        frag_charges = []
        frag_mults = []

        # Map atoms to indices
        for frag in fragments:
            indices = []
            frag_coords = frag.get_coords()

            for i in range(len(frag.symbols)):
                frag_coord = frag_coords[i]
                # Find matching atom in system
                for sys_idx, atom in enumerate(system.atoms):
                    if np.allclose(atom.coords, frag_coord, atol=1e-5):
                        indices.append(sys_idx)
                        break

            frag_indices.append(indices)
            frag_charges.append(frag.molecular_charge)
            frag_mults.append(frag.molecular_multiplicity)

        result["fragments"] = frag_indices
        result["fragment_charges"] = frag_charges
        result["fragment_multiplicities"] = frag_mults

    # Add provenance
    from autofragment._version import __version__

    result["provenance"] = {
        "creator": "autofragment",
        "version": __version__,
    }

    return result


def write_qcschema(
    system: ChemicalSystem,
    filepath: Union[str, Path],
    fragments: Optional[List[Fragment]] = None,
    name: Optional[str] = None,
    indent: int = 2,
) -> None:
    """
    Write ChemicalSystem to QCSchema JSON file.

    Parameters
    ----------
    system : ChemicalSystem
        The chemical system to write.
    filepath : str or Path
        Output file path.
    fragments : list, optional
        Fragment definitions to include.
    name : str, optional
        Name for the molecule.
    indent : int, optional
        JSON indentation level. Default is 2.

    Examples
    --------
    >>> from autofragment.core import ChemicalSystem, Atom
    >>> import numpy as np
    >>> atoms = [Atom(symbol="O", coords=np.array([0, 0, 0]))]
    >>> system = ChemicalSystem(atoms=atoms)
    >>> write_qcschema(system, "molecule.json")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = system_to_qcschema(system, fragments, name)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")


def write_qcschema_input(
    system: ChemicalSystem,
    filepath: Union[str, Path],
    driver: str = "energy",
    method: str = "hf",
    basis: str = "sto-3g",
    fragments: Optional[List[Fragment]] = None,
    keywords: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> None:
    """
    Write QCSchema input specification.

    Parameters
    ----------
    system : ChemicalSystem
        The chemical system.
    filepath : str or Path
        Output file path.
    driver : str, optional
        Calculation type (energy, gradient, hessian). Default is "energy".
    method : str, optional
        QC method. Default is "hf".
    basis : str, optional
        Basis set. Default is "sto-3g".
    fragments : list, optional
        Fragment definitions.
    keywords : dict, optional
        Additional keywords for the QC program.
    indent : int, optional
        JSON indentation level.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    molecule = system_to_qcschema(system, fragments)

    input_spec: Dict[str, Any] = {
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "molecule": molecule,
        "driver": driver,
        "model": {
            "method": method,
            "basis": basis,
        },
    }

    if keywords:
        input_spec["keywords"] = keywords

    with open(path, "w") as f:
        json.dump(input_spec, f, indent=indent)
        f.write("\n")
