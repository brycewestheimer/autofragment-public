# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for topology neighborhood selection utilities."""

import numpy as np
import pytest

from autofragment.partitioners.topology import TopologyNeighborSelection


def test_graph_hop_selection():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ])
    elements = ["C", "C", "C", "C"]
    bonds = [(0, 1), (1, 2), (2, 3)]

    selection = TopologyNeighborSelection(
        seed_atoms={0},
        mode="graph",
        hops=2,
        bond_policy="strict",
    )
    result = selection.select(coords, elements, bonds=bonds)

    assert result.selected_atoms == {0, 1, 2}
    assert result.shells[0] == {1}
    assert result.shells[1] == {2}


def test_euclidean_layer_selection():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    elements = ["C", "C", "C", "C"]

    selection = TopologyNeighborSelection(
        seed_atoms={0},
        mode="euclidean",
        layers=2,
        k_per_layer=1,
    )
    result = selection.select(coords, elements)

    assert result.selected_atoms == {0, 1, 2}


def test_residue_expansion():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    elements = ["C", "C", "C"]

    selection = TopologyNeighborSelection(
        seed_atoms={0},
        mode="graph",
        hops=0,
        expand_residues=True,
        bond_policy="strict",
    )
    result = selection.select(
        coords,
        elements,
        bonds=[(0, 1)],
        residue_numbers=[1, 1, 2],
    )

    assert result.selected_atoms == {0, 1}


def test_graph_strict_requires_bonds():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
    ])
    elements = ["C", "C"]

    selection = TopologyNeighborSelection(
        seed_atoms={0},
        mode="graph",
        hops=1,
        bond_policy="strict",
    )

    with pytest.raises(ValueError, match="requires explicit bonds"):
        selection.select(coords, elements)
