# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Topology-based neighborhood selection utilities.

This module provides reusable atom-neighborhood selection based on
bond-topology shells (graph hops) and Euclidean nearest-neighbor layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from autofragment.core.bonds import COVALENT_RADII

SelectionMode = Literal["graph", "euclidean"]
BondPolicy = Literal["infer", "strict"]


@dataclass(frozen=True)
class ShellSelectionResult:
    """Selected atoms and per-layer shell decomposition."""

    selected_atoms: Set[int]
    shells: List[Set[int]]


def _normalize_bonds(bonds: Sequence[Any], n_atoms: int) -> List[Tuple[int, int]]:
    """Normalize tuple/dict bond inputs into validated (i, j) tuples."""
    normalized: List[Tuple[int, int]] = []
    for bond in bonds:
        i: int
        j: int
        if isinstance(bond, tuple) and len(bond) == 2:
            i, j = int(bond[0]), int(bond[1])
        elif isinstance(bond, dict) and "atom1" in bond and "atom2" in bond:
            i, j = int(bond["atom1"]), int(bond["atom2"])
        else:
            continue

        if i == j:
            continue
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            continue
        if i < j:
            normalized.append((i, j))
        else:
            normalized.append((j, i))

    return sorted(set(normalized))


def _infer_bonds(
    coords: np.ndarray,
    elements: Sequence[str],
    tolerance: float = 0.4,
) -> List[Tuple[int, int]]:
    """Infer bonds by covalent-radii cutoff."""
    n_atoms = len(coords)
    inferred: List[Tuple[int, int]] = []
    for i in range(n_atoms):
        r_i = COVALENT_RADII.get(elements[i], 1.5)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII.get(elements[j], 1.5)
            cutoff = r_i + r_j + tolerance
            if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
                inferred.append((i, j))
    return inferred


def _build_adjacency(n_atoms: int, bonds: Sequence[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """Build undirected adjacency map from bond list."""
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def _expand_graph_shells(
    seed_atoms: Set[int],
    adjacency: Dict[int, Set[int]],
    hops: int,
) -> ShellSelectionResult:
    """Expand from seed atoms by graph-hop shells."""
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")

    visited = set(seed_atoms)
    frontier = set(seed_atoms)
    shells: List[Set[int]] = []

    for _ in range(hops):
        nxt: Set[int] = set()
        for atom_idx in frontier:
            nxt.update(adjacency.get(atom_idx, set()))
        nxt -= visited
        shells.append(nxt)
        visited.update(nxt)
        frontier = nxt
        if not frontier:
            break

    return ShellSelectionResult(selected_atoms=visited, shells=shells)


def _expand_euclidean_layers(
    seed_atoms: Set[int],
    coords: np.ndarray,
    layers: int,
    k_per_layer: int,
) -> ShellSelectionResult:
    """Expand from seed atoms by Euclidean nearest-neighbor layers."""
    if layers < 0:
        raise ValueError(f"layers must be >= 0, got {layers}")
    if k_per_layer < 0:
        raise ValueError(f"k_per_layer must be >= 0, got {k_per_layer}")

    selected = set(seed_atoms)
    shells: List[Set[int]] = []

    if layers == 0 or k_per_layer == 0:
        return ShellSelectionResult(selected_atoms=selected, shells=shells)

    n_atoms = len(coords)
    for _ in range(layers):
        remaining = [idx for idx in range(n_atoms) if idx not in selected]
        if not remaining:
            shells.append(set())
            continue

        scored: List[Tuple[float, int]] = []
        for idx in remaining:
            min_dist = min(float(np.linalg.norm(coords[idx] - coords[s])) for s in selected)
            scored.append((min_dist, idx))

        scored.sort(key=lambda item: (item[0], item[1]))
        layer_atoms = {idx for _, idx in scored[:k_per_layer]}
        shells.append(layer_atoms)
        selected.update(layer_atoms)

    return ShellSelectionResult(selected_atoms=selected, shells=shells)


def _expand_to_residues(
    selected_atoms: Set[int],
    residue_names: Optional[Sequence[Any]] = None,
    residue_numbers: Optional[Sequence[Any]] = None,
    chains: Optional[Sequence[Any]] = None,
) -> Set[int]:
    """Expand selected atoms to complete residue groups when metadata exists."""
    if not selected_atoms:
        return selected_atoms

    residue_names = residue_names or []
    residue_numbers = residue_numbers or []
    chains = chains or []

    max_len = max([len(residue_names), len(residue_numbers), len(chains)], default=0)
    if max_len == 0:
        return selected_atoms

    def _key(idx: int) -> Optional[Tuple[Any, Any, Any]]:
        """Internal helper to key."""
        name = residue_names[idx] if idx < len(residue_names) else None
        number = residue_numbers[idx] if idx < len(residue_numbers) else None
        chain = chains[idx] if idx < len(chains) else None
        if name is None and number is None and chain is None:
            return None
        return (name, number, chain)

    selected_keys: Set[Tuple[Any, Any, Any]] = set()
    for atom_idx in selected_atoms:
        key = _key(atom_idx)
        if key is not None:
            selected_keys.add(key)

    if not selected_keys:
        return selected_atoms

    expanded = set(selected_atoms)
    for idx in range(max_len):
        key = _key(idx)
        if key in selected_keys:
            expanded.add(idx)
    return expanded


@dataclass(frozen=True)
class TopologyNeighborSelection:
    """Reusable topology/neighbor shell selection configuration."""

    seed_atoms: Set[int]
    mode: SelectionMode = "graph"
    hops: int = 1
    layers: int = 1
    k_per_layer: int = 1
    expand_residues: bool = False
    bond_policy: BondPolicy = "infer"

    def select(
        self,
        coords: np.ndarray,
        elements: Sequence[str],
        *,
        bonds: Optional[Sequence[Any]] = None,
        residue_names: Optional[Sequence[Any]] = None,
        residue_numbers: Optional[Sequence[Any]] = None,
        chains: Optional[Sequence[Any]] = None,
    ) -> ShellSelectionResult:
        """Select atoms according to configured mode and neighborhood depth."""
        n_atoms = len(coords)
        if n_atoms == 0:
            return ShellSelectionResult(selected_atoms=set(), shells=[])
        if not self.seed_atoms:
            raise ValueError("seed_atoms must not be empty")

        seed_atoms = {int(i) for i in self.seed_atoms}
        invalid = [i for i in seed_atoms if i < 0 or i >= n_atoms]
        if invalid:
            raise ValueError(f"seed_atoms contains invalid atom indices: {invalid}")

        if self.mode == "graph":
            bond_tuples: List[Tuple[int, int]]
            if bonds is not None:
                bond_tuples = _normalize_bonds(bonds, n_atoms)
            else:
                bond_tuples = []

            if not bond_tuples and self.bond_policy == "infer":
                bond_tuples = _infer_bonds(coords, elements)

            if not bond_tuples and self.bond_policy == "strict":
                raise ValueError("Graph topology selection requires explicit bonds in strict mode")

            adjacency = _build_adjacency(n_atoms, bond_tuples)
            result = _expand_graph_shells(seed_atoms, adjacency, self.hops)
        else:
            result = _expand_euclidean_layers(seed_atoms, coords, self.layers, self.k_per_layer)

        if self.expand_residues:
            expanded = _expand_to_residues(
                result.selected_atoms,
                residue_names=residue_names,
                residue_numbers=residue_numbers,
                chains=chains,
            )
            return ShellSelectionResult(selected_atoms=expanded, shells=result.shells)

        return result
