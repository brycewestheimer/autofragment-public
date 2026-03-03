# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from autofragment.geometry.tilings import get_tiling_shape, list_tiling_shapes


def _volume_from_vectors(vectors: np.ndarray) -> float:
    return abs(np.linalg.det(vectors))


def test_registry_contains_all_shapes():
    names = set(list_tiling_shapes())
    assert {
        "cube",
        "hex_prism",
        "truncated_octahedron",
        "elongated_dodecahedron",
        "rhombic_dodecahedron",
    }.issubset(names)


def test_cube_tiling_geometry():
    shape = get_tiling_shape("cube")
    scale = 2.0
    vectors = shape.lattice_vectors(scale)
    assert np.isclose(_volume_from_vectors(vectors), scale ** 3)
    assert shape.point_in_cell(np.array([0.0, 0.0, 0.0]), scale)
    assert shape.point_in_cell(np.array([scale / 2.0, 0.0, 0.0]), scale)
    assert not shape.point_in_cell(np.array([scale, 0.0, 0.0]), scale)


def test_hexagonal_prism_geometry():
    shape = get_tiling_shape("hex_prism")
    scale = 3.0
    vectors = shape.lattice_vectors(scale)
    expected_volume = np.sqrt(3) / 2.0 * scale ** 3
    assert np.isclose(_volume_from_vectors(vectors), expected_volume)
    assert shape.point_in_cell(np.array([0.0, 0.0, 0.0]), scale)
    side = scale / np.sqrt(3.0)
    assert shape.point_in_cell(np.array([side / 2.0, 0.0, 0.0]), scale)
    assert not shape.point_in_cell(np.array([side * 2.0, 0.0, 0.0]), scale)


def test_truncated_octahedron_geometry():
    shape = get_tiling_shape("truncated_octahedron")
    scale = 2.0
    vectors = shape.lattice_vectors(scale)
    assert np.isclose(_volume_from_vectors(vectors), scale ** 3 / 2.0)
    assert shape.point_in_cell(np.array([0.0, 0.0, 0.0]), scale)
    for vertex in shape.vertex_set(scale):
        assert shape.point_in_cell(vertex, scale)
    assert not shape.point_in_cell(np.array([scale, 0.0, 0.0]), scale)


def test_rhombic_dodecahedron_geometry():
    shape = get_tiling_shape("rhombic_dodecahedron")
    scale = 2.0
    vectors = shape.lattice_vectors(scale)
    assert np.isclose(_volume_from_vectors(vectors), scale ** 3 / 4.0)
    assert shape.point_in_cell(np.array([0.0, 0.0, 0.0]), scale)
    for vertex in shape.vertex_set(scale):
        assert shape.point_in_cell(vertex, scale)
    assert not shape.point_in_cell(np.array([scale, 0.0, 0.0]), scale)


def test_elongated_dodecahedron_geometry():
    shape = get_tiling_shape("elongated_dodecahedron")
    scale = 2.0
    vectors = shape.lattice_vectors(scale)
    expected_volume = scale ** 3 / 4.0 * shape.elongation
    assert np.isclose(_volume_from_vectors(vectors), expected_volume)
    assert shape.point_in_cell(np.array([0.0, 0.0, 0.0]), scale)
    inside = np.array([0.0, 0.0, scale * shape.elongation / 4.0])
    outside = np.array([0.0, 0.0, scale * shape.elongation])
    assert shape.point_in_cell(inside, scale)
    assert not shape.point_in_cell(outside, scale)
