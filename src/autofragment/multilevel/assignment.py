# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Layer assignment algorithms for multi-level methods.

This module provides algorithms for automatically assigning atoms to
computational layers based on various criteria such as distance,
residue membership, or custom selectors.

Example:
    >>> from autofragment.multilevel.assignment import assign_by_distance
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0]])
    >>> layers = assign_by_distance(coords, center_atoms={0}, layer_cutoffs=[3.0])
    >>> layers  # [{0}, {1}, {2}] - center, within 3A, beyond 3A
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


def compute_min_distance_to_atoms(
    coords: np.ndarray,
    atom_index: int,
    reference_atoms: Set[int],
) -> float:
    """Compute minimum distance from an atom to a set of reference atoms.

    Args:
        coords: (N, 3) array of coordinates.
        atom_index: Index of the atom to measure from.
        reference_atoms: Set of reference atom indices.

    Returns:
        Minimum distance to any reference atom.
    """
    if not reference_atoms:
        return float("inf")

    atom_coord = coords[atom_index]
    min_dist = float("inf")

    for ref_idx in reference_atoms:
        dist = float(np.linalg.norm(atom_coord - coords[ref_idx]))
        if dist < min_dist:
            min_dist = dist

    return min_dist


def assign_by_distance(
    coords: np.ndarray,
    center_atoms: Set[int],
    layer_cutoffs: List[float],
) -> List[Set[int]]:
    """Assign atoms to layers based on distance from center atoms.

    Creates multiple layers based on distance cutoffs from a set of
    center atoms. Layer 0 contains the center atoms, layer 1 contains
    atoms within the first cutoff, etc.

    Args:
        coords: (N, 3) array of atomic coordinates in Angstroms.
        center_atoms: Atom indices defining the center (innermost layer).
        layer_cutoffs: Distance cutoffs [r1, r2, ...] for each layer boundary.
            Atoms within r1 of center go to layer 1, within r2 to layer 2, etc.

    Returns:
        List of atom index sets, one per layer. Layer 0 is the center,
        subsequent layers are sorted by increasing distance.

    Example:
        >>> coords = np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0], [10, 0, 0]])
        >>> layers = assign_by_distance(coords, {0}, [3.0, 7.0])
        >>> layers
        [{0}, {1}, {2}, {3}]  # center, <3A, 3-7A, >7A
    """
    n_atoms = len(coords)
    n_layers = len(layer_cutoffs) + 1

    # Initialize layers: layer 0 is center, others start empty
    layers: List[Set[int]] = [set() for _ in range(n_layers)]
    layers[0] = set(center_atoms)

    for atom_idx in range(n_atoms):
        if atom_idx in center_atoms:
            continue

        # Compute minimum distance to any center atom
        min_dist = compute_min_distance_to_atoms(coords, atom_idx, center_atoms)

        # Determine layer based on cutoffs
        # Default to outermost layer
        assigned_layer = n_layers - 1

        for layer_idx, cutoff in enumerate(layer_cutoffs):
            if min_dist <= cutoff:
                assigned_layer = layer_idx + 1  # +1 because layer 0 is center
                break

        layers[assigned_layer].add(atom_idx)

    return layers


def assign_by_residue(
    residue_ids: List[str],
    qm_residues: List[str],
) -> Tuple[Set[int], Set[int]]:
    """Assign atoms to QM or MM region based on residue name.

    Args:
        residue_ids: List of residue identifiers for each atom.
            Can include residue name and number, e.g., ["ALA1", "GLY2", "WAT1"].
        qm_residues: Residue identifiers that should be in QM region.

    Returns:
        Tuple of (qm_atoms, mm_atoms) as sets of atom indices.

    Example:
        >>> residues = ["ALA", "ALA", "GLY", "WAT", "WAT"]
        >>> qm, mm = assign_by_residue(residues, ["ALA", "GLY"])
        >>> qm  # {0, 1, 2}
        >>> mm  # {3, 4}
    """
    qm_residue_set = set(qm_residues)
    qm_atoms: Set[int] = set()
    mm_atoms: Set[int] = set()

    for i, residue in enumerate(residue_ids):
        if residue in qm_residue_set:
            qm_atoms.add(i)
        else:
            mm_atoms.add(i)

    return qm_atoms, mm_atoms


def assign_by_residue_number(
    residue_numbers: List[int],
    qm_residue_numbers: List[int],
) -> Tuple[Set[int], Set[int]]:
    """Assign atoms to QM or MM region based on residue number.

    Args:
        residue_numbers: List of residue numbers for each atom.
        qm_residue_numbers: Residue numbers that should be in QM region.

    Returns:
        Tuple of (qm_atoms, mm_atoms) as sets of atom indices.
    """
    qm_numbers_set = set(qm_residue_numbers)
    qm_atoms: Set[int] = set()
    mm_atoms: Set[int] = set()

    for i, resnum in enumerate(residue_numbers):
        if resnum in qm_numbers_set:
            qm_atoms.add(i)
        else:
            mm_atoms.add(i)

    return qm_atoms, mm_atoms


def assign_by_element(
    elements: List[str],
    qm_elements: List[str],
) -> Tuple[Set[int], Set[int]]:
    """Assign atoms to QM or MM region based on element type.

    This is useful for simple systems where specific elements should
    be treated quantum mechanically (e.g., transition metals).

    Args:
        elements: List of element symbols for each atom.
        qm_elements: Element symbols that should be in QM region.

    Returns:
        Tuple of (qm_atoms, mm_atoms) as sets of atom indices.

    Example:
        >>> elements = ["Fe", "N", "N", "N", "N", "C", "C", "C"]
        >>> qm, mm = assign_by_element(elements, ["Fe", "N"])
        >>> qm  # {0, 1, 2, 3, 4}
    """
    qm_element_set = set(qm_elements)
    qm_atoms: Set[int] = set()
    mm_atoms: Set[int] = set()

    for i, element in enumerate(elements):
        if element in qm_element_set:
            qm_atoms.add(i)
        else:
            mm_atoms.add(i)

    return qm_atoms, mm_atoms


def assign_by_custom(
    n_atoms: int,
    selector: Callable[[int], str],
) -> Dict[str, Set[int]]:
    """Assign atoms to layers using a custom selector function.

    This provides maximum flexibility for layer assignment by
    allowing any user-defined logic.

    Args:
        n_atoms: Total number of atoms.
        selector: Function that takes an atom index and returns a layer name.

    Returns:
        Dict mapping layer names to sets of atom indices.

    Example:
        >>> def my_selector(atom_idx):
        ...     if atom_idx < 5:
        ...         return "qm"
        ...     elif atom_idx < 20:
        ...         return "buffer"
        ...     else:
        ...         return "mm"
        >>> layers = assign_by_custom(100, my_selector)
    """
    layers: Dict[str, Set[int]] = {}

    for atom_idx in range(n_atoms):
        layer_name = selector(atom_idx)
        if layer_name not in layers:
            layers[layer_name] = set()
        layers[layer_name].add(atom_idx)

    return layers


def assign_by_atom_properties(
    atom_properties: List[Dict[str, Any]],
    selector: Callable[[Dict[str, Any]], str],
) -> Dict[str, Set[int]]:
    """Assign atoms to layers based on atom properties.

    Args:
        atom_properties: List of property dicts for each atom.
            Could contain "element", "residue", "chain", etc.
        selector: Function that takes atom properties and returns layer name.

    Returns:
        Dict mapping layer names to sets of atom indices.

    Example:
        >>> props = [
        ...     {"element": "Fe", "residue": "HEM"},
        ...     {"element": "N", "residue": "HIS"},
        ... ]
        >>> def selector(p):
        ...     return "qm" if p["residue"] in ["HEM", "HIS"] else "mm"
        >>> layers = assign_by_atom_properties(props, selector)
    """
    layers: Dict[str, Set[int]] = {}

    for atom_idx, props in enumerate(atom_properties):
        layer_name = selector(props)
        if layer_name not in layers:
            layers[layer_name] = set()
        layers[layer_name].add(atom_idx)

    return layers


def expand_selection_to_residues(
    atom_selection: Set[int],
    atom_to_residue: List[int],
) -> Set[int]:
    """Expand atom selection to include complete residues.

    When atoms are selected (e.g., by distance), this function expands
    the selection to include all atoms in the same residues. This ensures
    layer boundaries don't cut through residues.

    Args:
        atom_selection: Initial set of selected atom indices.
        atom_to_residue: Mapping from atom index to residue number.

    Returns:
        Expanded set of atom indices including complete residues.
    """
    # Find all residues represented in the selection
    selected_residues = {atom_to_residue[i] for i in atom_selection}

    # Find all atoms in those residues
    expanded_selection: Set[int] = set()
    for atom_idx, resnum in enumerate(atom_to_residue):
        if resnum in selected_residues:
            expanded_selection.add(atom_idx)

    return expanded_selection


def validate_layer_assignment(
    layers: List[Set[int]],
    n_atoms: int,
) -> Tuple[bool, Optional[str]]:
    """Validate that a layer assignment covers all atoms exactly once.

    Args:
        layers: List of atom index sets.
        n_atoms: Total number of atoms expected.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check for completeness
    all_assigned = set()
    for layer in layers:
        all_assigned.update(layer)

    expected = set(range(n_atoms))
    missing = expected - all_assigned
    if missing:
        return False, f"Atoms {missing} are not assigned to any layer"

    # Check for overlaps
    seen: Set[int] = set()
    for i, layer in enumerate(layers):
        overlap = seen & layer
        if overlap:
            return False, f"Atoms {overlap} assigned to multiple layers (layer {i})"
        seen.update(layer)

    return True, None
