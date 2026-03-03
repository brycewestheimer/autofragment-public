# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Hypothesis strategies for molecular systems."""
import numpy as np
from hypothesis import strategies as st

from autofragment.core.graph import MolecularGraph
from autofragment.core.types import Atom

# Element strategy
elements = st.sampled_from(["C", "H", "O", "N", "S", "P", "F", "Cl"])

# Position strategy
positions = st.tuples(
    st.floats(-100, 100),
    st.floats(-100, 100),
    st.floats(-100, 100)
)

@st.composite
def atoms(draw):
    """Generate a random atom."""
    element = draw(elements)
    pos = draw(positions)
    return Atom(symbol=element, coords=np.array(pos))


@st.composite
def molecular_graphs(draw, min_atoms=1, max_atoms=50):
    """Generate a random molecular graph."""
    n_atoms = draw(st.integers(min_atoms, max_atoms))
    atom_list = [draw(atoms()) for _ in range(n_atoms)]

    graph = MolecularGraph()
    for i, atom in enumerate(atom_list):
        graph.add_atom(i, atom.symbol, atom.coords)

    # Add some random bonds
    if n_atoms > 1:
        # Max edges for simple graph is n(n-1)/2
        max_edges = min(n_atoms * 2, n_atoms * (n_atoms - 1) // 2)
        n_bonds = draw(st.integers(0, max_edges))

        seen_bonds = set()
        for _ in range(n_bonds):
            i = draw(st.integers(0, n_atoms - 1))
            j = draw(st.integers(0, n_atoms - 1))

            if i != j:
                u, v = min(i, j), max(i, j)
                if (u, v) not in seen_bonds:
                    seen_bonds.add((u, v))
                    # Check if bond exists currently?
                    # The set handles duplicates in this loop, but graph.add_bond might be idempotent or throw if dup?
                    # MolecularGraph uses networkx usually.
                    graph.add_bond(u, v)

    return graph


@st.composite
def connected_graphs(draw, min_atoms=2, max_atoms=30):
    """Generate connected molecular graphs."""
    n_atoms = draw(st.integers(min_atoms, max_atoms))

    graph = MolecularGraph()
    # Add first atom
    first_atom = draw(atoms())
    graph.add_atom(0, first_atom.symbol, first_atom.coords)

    for i in range(1, n_atoms):
        atom = draw(atoms())
        graph.add_atom(i, atom.symbol, atom.coords)

        # Connect to a previous atom to ensure connectivity (spanning tree construction)
        prev = draw(st.integers(0, i - 1))
        graph.add_bond(prev, i)

    return graph
