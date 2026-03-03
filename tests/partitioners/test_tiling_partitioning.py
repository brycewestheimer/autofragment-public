# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from autofragment.core.lattice import Lattice
from autofragment.core.types import Atom, ChemicalSystem
from autofragment.geometry.tilings import get_tiling_shape, grid_shape_from_count
from autofragment.partitioners.geometric import TilingPartitioner


def _generate_centers(bounds: np.ndarray, lattice_vectors: np.ndarray, grid_shape):
    origin = bounds[0]
    centers = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                centers.append(
                    origin
                    + (i + 0.5) * lattice_vectors[0]
                    + (j + 0.5) * lattice_vectors[1]
                    + (k + 0.5) * lattice_vectors[2]
                )
    return np.array(centers)


def _build_periodic_system(shape_name: str, n_fragments: int, scale: float) -> ChemicalSystem:
    shape = get_tiling_shape(shape_name)
    grid_shape = grid_shape_from_count(n_fragments)
    lattice_vectors = np.eye(3) * (scale * grid_shape[0])
    lattice = Lattice(lattice_vectors)
    bounds = np.vstack([[0.0, 0.0, 0.0], lattice_vectors.sum(axis=0)])
    centers = _generate_centers(bounds, shape.lattice_vectors(scale), grid_shape)
    atoms = [Atom(symbol="H", coords=center) for center in centers]
    return ChemicalSystem(atoms=atoms, lattice=lattice)


def test_periodic_tiling_partitioning_multiple_shapes():
    for shape_name in ("cube", "truncated_octahedron"):
        system = _build_periodic_system(shape_name, n_fragments=8, scale=1.0)
        partitioner = TilingPartitioner(
            tiling_shape=shape_name,
            n_fragments=8,
            scale=1.0,
        )
        tree = partitioner.partition(system)
        assert len(tree.fragments) == 8
        assert tree.partitioning["tiling_shape"] == shape_name


def test_nonperiodic_tiling_partitioning():
    shape_name = "cube"
    shape = get_tiling_shape(shape_name)
    grid_shape = grid_shape_from_count(8)
    scale = 1.0
    bounds = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    centers = _generate_centers(bounds, shape.lattice_vectors(scale), grid_shape)
    atoms = [Atom(symbol="He", coords=center) for center in centers]
    system = ChemicalSystem(atoms=atoms)
    partitioner = TilingPartitioner(
        tiling_shape=shape_name,
        n_fragments=8,
        scale=scale,
        bounds_strategy="atoms",
    )
    tree = partitioner.partition(system)
    assert len(tree.fragments) == 8
    for fragment in tree.fragments:
        assert fragment.metadata["tiling_shape"] == shape_name
