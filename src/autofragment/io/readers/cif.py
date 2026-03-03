# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
CIF (Crystallographic Information File) reader.

This module provides functions for reading CIF files into ChemicalSystem objects.
Uses gemmi library if available, otherwise falls back to basic parsing.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from autofragment.core.lattice import Lattice
from autofragment.core.types import Atom, ChemicalSystem

# Check for gemmi availability
try:
    import gemmi
    _HAS_GEMMI = True
except ImportError:
    _HAS_GEMMI = False


def read_cif(
    filepath: Union[str, Path],
    block_index: int = 0,
    use_gemmi: bool = True,
) -> ChemicalSystem:
    """
    Read CIF file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to CIF file.
    block_index : int, optional
        Which data block to read (0-indexed). Default is 0.
    use_gemmi : bool, optional
        Use gemmi library if available. Default is True.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.

    Notes
    -----
    If gemmi is available and use_gemmi=True, it will be used for
    robust parsing. Otherwise, a basic parser is used that may not
    handle all CIF features.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {filepath}")

    if _HAS_GEMMI and use_gemmi:
        return _read_cif_gemmi(path, block_index)
    else:
        return _read_cif_basic(path, block_index)


def _read_cif_gemmi(filepath: Path, block_index: int) -> ChemicalSystem:
    """
    Read CIF using gemmi library.

    Parameters
    ----------
    filepath : Path
        Path to CIF file.
    block_index : int
        Which data block to read.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.
    """
    doc = gemmi.cif.read(str(filepath))

    if block_index >= len(doc):
        raise IndexError(
            f"Block index {block_index} out of range "
            f"(file contains {len(doc)} blocks)"
        )

    block = doc[block_index]
    atoms = []
    metadata: Dict[str, Any] = {
        "source_format": "cif",
        "source_file": str(filepath.name),
        "block_name": block.name,
        "has_gemmi": True,
    }

    lattice = None

    # Try high-level structure parsing first
    try:
        # For macromolecular structures
        structure = gemmi.make_structure_from_block(block)

        for model in structure:
            for chain in model:
                for residue in chain:
                    for gemmi_atom in residue:
                        atoms.append(Atom(
                            symbol=gemmi_atom.element.name,
                            coords=np.array([gemmi_atom.pos.x, gemmi_atom.pos.y, gemmi_atom.pos.z]),
                        ))
            break  # Only first model

        if structure.cell.is_crystal():
            c = structure.cell
            lattice = Lattice.from_parameters(
                c.a, c.b, c.c, c.alpha, c.beta, c.gamma
            )
            metadata["cell"] = {
                "a": c.a, "b": c.b, "c": c.c,
                "alpha": c.alpha, "beta": c.beta, "gamma": c.gamma,
            }

    except Exception as e:
        # Fallback to manual parsing below
        warnings.warn(
            f"Gemmi high-level CIF parse failed, falling back to manual loop parsing: {e}",
            UserWarning,
            stacklevel=2,
        )

    # If high-level parsing failed or returned no atoms, try manual loop parsing
    if not atoms:
        # Identify relevant tags in the loop
        possible_tags = {
            "label": ["_atom_site_label", "_atom_site_id"],
            "element": ["_atom_site_type_symbol"],
            "x": ["_atom_site_Cartn_x", "_atom_site_fract_x"],
            "y": ["_atom_site_Cartn_y", "_atom_site_fract_y"],
            "z": ["_atom_site_Cartn_z", "_atom_site_fract_z"]
        }

        # Find the loop containing atom sites
        loop = None
        for key in ["label", "element"]:
            for tag in possible_tags[key]:
                col = block.find_loop(tag)
                if col:
                    loop = col.get_loop()
                    break
            if loop:
                break

        if loop:
            # Map valid tags to column indices
            col_map: Dict[str, int | str] = {}
            for tag in loop.tags:
                tag_lower = tag.lower()
                for key, options in possible_tags.items():
                    for opt in options:
                        if opt.lower() == tag_lower:
                            col_map[key] = loop.tags.index(tag)
                            col_map[f"{key}_type"] = "fract" if "fract" in opt.lower() else "cart"
                            break

            # Check we have minimum requirements
            if "x" in col_map and "y" in col_map and "z" in col_map:
                is_fract = col_map.get("x_type") == "fract"

                # Get cell if needed
                cell_params = _get_gemmi_cell(block) if is_fract else None
                if cell_params:
                    metadata["cell"] = cell_params
                    lattice = Lattice.from_parameters(
                        cell_params["a"], cell_params["b"], cell_params["c"],
                        cell_params["alpha"], cell_params["beta"], cell_params["gamma"]
                    )

                for i in range(loop.length()):
                    try:
                        # Extract element/symbol
                        element = "C" # Default
                        if "element" in col_map:
                            element = loop[i, cast(int, col_map["element"])]
                        elif "label" in col_map:
                            element = loop[i, cast(int, col_map["label"])]

                        # Clean element
                        element = re.sub(r'\d+', '', element).strip()
                        if len(element) > 2:
                            element = element[:2]

                        x = float(loop[i, cast(int, col_map["x"])])
                        y = float(loop[i, cast(int, col_map["y"])])
                        z = float(loop[i, cast(int, col_map["z"])])

                        atoms.append(Atom(
                            symbol=element,
                            coords=np.array([x, y, z])
                        ))
                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            f"Malformed CIF atom_site row at index {i} in block {block.name!r}"
                        ) from e

                # If fractional, convert to Cartesian
                if is_fract and lattice:
                     # Use our Lattice object to convert
                     coords = np.array([a.coords for a in atoms])
                     cart_coords = lattice.fractional_to_cartesian(coords)
                     for i, parsed_atom in enumerate(atoms):
                         parsed_atom.coords = cart_coords[i]

    return ChemicalSystem(atoms=atoms, bonds=[], metadata=metadata, lattice=lattice)


def _get_gemmi_cell(block) -> Optional[Dict[str, float]]:
    """Helper to extract cell from gemmi block."""
    try:
        a = float(block.find_value('_cell_length_a'))
        b = float(block.find_value('_cell_length_b'))
        c = float(block.find_value('_cell_length_c'))
        alpha = float(block.find_value('_cell_angle_alpha'))
        beta = float(block.find_value('_cell_angle_beta'))
        gamma = float(block.find_value('_cell_angle_gamma'))
        return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    except (ValueError, TypeError):
        return None


def _read_cif_basic(filepath: Path, block_index: int) -> ChemicalSystem:
    """
    Read CIF using basic parser (no gemmi).

    This is a simplified parser that handles common CIF files but may
    not work for all cases.

    Parameters
    ----------
    filepath : Path
        Path to CIF file.
    block_index : int
        Which data block to read (0 = first).

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms.
    """
    content = filepath.read_text()

    # Find data blocks
    blocks = re.split(r'data_', content)[1:]  # Split and skip empty first

    if not blocks:
        raise ValueError(f"No data blocks found in CIF file: {filepath}")

    if block_index >= len(blocks):
        raise IndexError(
            f"Block index {block_index} out of range "
            f"(file contains {len(blocks)} blocks)"
        )

    block_content = blocks[block_index]
    block_name = block_content.split()[0] if block_content.split() else ""

    # Find atom_site loop
    atoms = _parse_atom_site_loop(block_content)

    # Get cell parameters if present
    cell = _parse_cell_parameters(block_content)

    # If coordinates are fractional and we have cell, convert to Cartesian
    if cell and atoms:
        # Check if coordinates look fractional (all between 0 and 1)
        coords = np.array([a.coords for a in atoms])
        if np.all(coords >= -0.1) and np.all(coords <= 1.1):
            # Likely fractional, convert
            atoms = _convert_fractional_to_cartesian(atoms, cell)

    metadata: Dict[str, Any] = {
        "source_format": "cif",
        "source_file": str(filepath.name),
        "block_name": block_name,
        "has_gemmi": False,
    }

    lattice = None
    if cell:
        metadata["cell"] = cell
        lattice = Lattice.from_parameters(
            cell["a"], cell["b"], cell["c"],
            cell["alpha"], cell["beta"], cell["gamma"],
        )

    return ChemicalSystem(atoms=atoms, bonds=[], metadata=metadata, lattice=lattice)


def _parse_atom_site_loop(content: str) -> List[Atom]:
    """
    Parse atom_site loop from CIF content.

    Parameters
    ----------
    content : str
        CIF block content.

    Returns
    -------
    list
        List of Atom objects.
    """
    atoms = []

    # Find loop_ containing atom_site
    loop_pattern = r'loop_\s*((?:_atom_site_\w+\s*)+)((?:[^\n_loop][^\n]*\n?)*)'

    for match in re.finditer(loop_pattern, content, re.IGNORECASE):
        columns_text = match.group(1)
        data_text = match.group(2)

        if "_atom_site" not in columns_text.lower():
            continue

        # Parse column names
        columns = [c.strip().lower() for c in columns_text.split() if c.strip()]

        # Find relevant column indices
        elem_idx = None
        label_idx = None
        x_idx = None
        y_idx = None
        z_idx = None

        for i, col in enumerate(columns):
            if col == "_atom_site_type_symbol":
                elem_idx = i
            elif col == "_atom_site_label":
                label_idx = i
            elif col in ("_atom_site_cartn_x", "_atom_site_fract_x"):
                x_idx = i
            elif col in ("_atom_site_cartn_y", "_atom_site_fract_y"):
                y_idx = i
            elif col in ("_atom_site_cartn_z", "_atom_site_fract_z"):
                z_idx = i

        if x_idx is None or y_idx is None or z_idx is None:
            continue

        if elem_idx is None and label_idx is None:
            continue

        # Parse data rows
        for line in data_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("_"):
                continue

            # Handle quoted values
            parts = _split_cif_line(line)

            if len(parts) < max(x_idx, y_idx, z_idx) + 1:
                continue

            try:
                # Get element
                if elem_idx is not None and parts[elem_idx] != "?":
                    element = parts[elem_idx]
                elif label_idx is not None:
                    element = parts[label_idx][:2].strip()
                else:
                    continue

                # Clean element symbol
                element = re.sub(r'[0-9\+\-]', '', element).strip()
                if len(element) > 2:
                    element = element[:2]
                element = element.capitalize()

                # Get coordinates (remove uncertainty in parentheses)
                x = float(re.sub(r'\([^)]+\)', '', parts[x_idx]))
                y = float(re.sub(r'\([^)]+\)', '', parts[y_idx]))
                z = float(re.sub(r'\([^)]+\)', '', parts[z_idx]))

                atoms.append(Atom(
                    symbol=element,
                    coords=np.array([x, y, z]),
                ))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Malformed CIF atom_site row: {line!r}") from e

        break  # Only use first atom_site loop

    return atoms


def _split_cif_line(line: str) -> List[str]:
    """
    Split CIF data line handling quoted values.

    Parameters
    ----------
    line : str
        A data line from a CIF loop.

    Returns
    -------
    list
        List of values.
    """
    parts = []
    current = ""
    in_quote = False
    quote_char = None

    for char in line:
        if char in ("'", '"') and not in_quote:
            in_quote = True
            quote_char = char
        elif char == quote_char and in_quote:
            in_quote = False
            quote_char = None
        elif char.isspace() and not in_quote:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char

    if current:
        parts.append(current)

    return parts


def _parse_cell_parameters(content: str) -> Optional[Dict[str, float]]:
    """
    Parse cell parameters from CIF content.

    Parameters
    ----------
    content : str
        CIF block content.

    Returns
    -------
    dict or None
        Cell parameters {a, b, c, alpha, beta, gamma} or None.
    """
    cell: Dict[str, float] = {}

    param_patterns = [
        (r'_cell_length_a\s+([\d.]+)', "a"),
        (r'_cell_length_b\s+([\d.]+)', "b"),
        (r'_cell_length_c\s+([\d.]+)', "c"),
        (r'_cell_angle_alpha\s+([\d.]+)', "alpha"),
        (r'_cell_angle_beta\s+([\d.]+)', "beta"),
        (r'_cell_angle_gamma\s+([\d.]+)', "gamma"),
    ]

    for pattern, key in param_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                cell[key] = float(match.group(1))
            except ValueError:
                warnings.warn(
                    f"Skipping malformed CIF cell parameter {key}: {match.group(1)!r}",
                    UserWarning,
                    stacklevel=2,
                )

    if len(cell) == 6:
        return cell

    return None


def _convert_fractional_to_cartesian(
    atoms: List[Atom],
    cell: Dict[str, float],
) -> List[Atom]:
    """
    Convert fractional coordinates to Cartesian.

    Parameters
    ----------
    atoms : list
        Atoms with fractional coordinates.
    cell : dict
        Cell parameters.

    Returns
    -------
    list
        Atoms with Cartesian coordinates.
    """
    # Build transformation matrix
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    alpha = np.radians(cell["alpha"])
    beta = np.radians(cell["beta"])
    gamma = np.radians(cell["gamma"])

    # Cell vectors
    # a along x
    va = np.array([a, 0, 0])

    # b in xy plane
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

    # c from angles
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    vc = np.array([cx, cy, cz])

    # Transformation matrix
    M = np.column_stack([va, vb, vc])

    # Convert atoms
    new_atoms = []
    for atom in atoms:
        cart = M @ atom.coords
        new_atoms.append(Atom(
            symbol=atom.symbol,
            coords=cart,
            charge=atom.charge,
        ))

    return new_atoms
