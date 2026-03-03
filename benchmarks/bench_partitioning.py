# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Performance benchmarks for partitioning."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from autofragment.core.types import Atom, Molecule, molecules_to_system
from autofragment.partitioners.molecular import MolecularPartitioner


def _water_molecule(offset: np.ndarray) -> Molecule:
    oxygen = Atom(symbol="O", coords=offset)
    hydrogen1 = Atom(symbol="H", coords=offset + np.array([0.96, 0.0, 0.0]))
    hydrogen2 = Atom(symbol="H", coords=offset + np.array([-0.24, 0.93, 0.0]))
    return [oxygen, hydrogen1, hydrogen2]


def _water_cluster(n_molecules: int) -> List[Molecule]:
    molecules: List[Molecule] = []
    for i in range(n_molecules):
        offset = np.array([i * 3.0, 0.0, 0.0])
        molecules.append(_water_molecule(offset))
    return molecules


@pytest.fixture
def medium_system():
    molecules = _water_cluster(333)
    return molecules_to_system(molecules)


@pytest.fixture
def large_system():
    molecules = _water_cluster(3333)
    return molecules_to_system(molecules)


def test_partition_1k_atoms(benchmark, medium_system):
    """Benchmark partitioning ~1000 atoms."""
    partitioner = MolecularPartitioner(n_fragments=10, method="kmeans")
    result = benchmark(partitioner.partition, medium_system)
    assert len(result.fragments) > 0


def test_partition_10k_atoms(benchmark, large_system):
    """Benchmark partitioning ~10000 atoms."""
    partitioner = MolecularPartitioner(n_fragments=40, method="kmeans")
    result = benchmark(partitioner.partition, large_system)
    assert len(result.fragments) > 0
