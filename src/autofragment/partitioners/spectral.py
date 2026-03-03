# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Spectral partitioner with optional graph dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from autofragment.core.types import ChemicalSystem, Fragment, FragmentTree
from autofragment.optional import require_dependency
from autofragment.partitioners.base import BasePartitioner


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for spectral partitioning."""

    n_fragments: int = 2
    random_state: int = 42


class SpectralPartitioner(BasePartitioner):
    """Partitioner that uses spectral cuts on a molecular graph."""

    def __init__(self, n_fragments: int = 2, random_state: int = 42):
        """Initialize a new SpectralPartitioner instance."""
        if n_fragments <= 0:
            raise ValueError("n_fragments must be positive")
        self.config = SpectralConfig(n_fragments=n_fragments, random_state=random_state)

    def partition(
        self,
        system: ChemicalSystem,
        source_file: str | None = None,
    ) -> FragmentTree:
        """Partition a chemical system into fragments using spectral clustering."""
        nx = require_dependency("networkx", "graph", "Spectral partitioning")
        _ = require_dependency("scipy", "graph", "Spectral partitioning")

        graph = system.to_graph()

        partitions = _spectral_recursive_partition(nx, graph._graph, self.config.n_fragments)
        fragments = _fragments_from_partitions(system, partitions)

        return FragmentTree(
            fragments=fragments,
            source={"file": source_file} if source_file else {},
            partitioning={"algorithm": "spectral", "n_fragments": self.config.n_fragments},
        )


def _spectral_recursive_partition(nx, graph, n_parts: int) -> List[List[int]]:
    """Recursively bisect a graph using the Fiedler vector."""
    if n_parts <= 1 or graph.number_of_nodes() == 0:
        return [list(graph.nodes())]

    if graph.number_of_nodes() == 1:
        return [list(graph.nodes())]

    laplacian = nx.laplacian_matrix(graph).astype(float)
    _, vectors = _fiedler(laplacian)
    split_vector = vectors[:, 1]
    left = [node for node, value in zip(graph.nodes(), split_vector) if value <= 0]
    right = [node for node, value in zip(graph.nodes(), split_vector) if value > 0]

    if not left or not right:
        nodes = list(graph.nodes())
        mid = len(nodes) // 2
        left, right = nodes[:mid], nodes[mid:]

    parts_left = max(1, int(round(n_parts * len(left) / len(graph.nodes()))))
    parts_left = min(parts_left, n_parts - 1)
    parts_right = n_parts - parts_left

    partitions = []
    partitions.extend(_spectral_recursive_partition(nx, graph.subgraph(left), parts_left))
    partitions.extend(_spectral_recursive_partition(nx, graph.subgraph(right), parts_right))
    return partitions


def _fiedler(laplacian):
    """Return eigenvalues/vectors for the two smallest eigenvalues."""
    from scipy.sparse.linalg import eigsh

    eigenvalues, eigenvectors = eigsh(laplacian, k=2, which="SM")
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _fragments_from_partitions(
    system: ChemicalSystem,
    partitions: List[List[int]],
) -> List[Fragment]:
    """Build Fragment objects from atom index partitions."""
    atoms = system.atoms

    fragments: List[Fragment] = []
    for index, atom_indices in enumerate(partitions, start=1):
        symbols = [atoms[i].symbol for i in atom_indices]
        geometry = [coord for i in atom_indices for coord in atoms[i].coords.tolist()]
        fragments.append(
            Fragment(
                id=f"F{index}",
                symbols=symbols,
                geometry=geometry,
                metadata={"atom_indices": atom_indices},
            )
        )
    return fragments
