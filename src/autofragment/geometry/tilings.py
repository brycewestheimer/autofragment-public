# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tiling shapes and registry for geometric partitioning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TilingCell:
    """Container for tiling cell metadata.

    Attributes
    ----------
    name : str
        Registered tiling name.
    lattice_vectors : np.ndarray
        Translation vectors defining the tiling lattice (shape (3, 3)).
    cell_volume : float
        Volume of the fundamental cell.
    parameters : dict
        Additional parameters used to define the cell.
    """

    name: str
    lattice_vectors: np.ndarray
    cell_volume: float
    parameters: dict = field(default_factory=dict)


class TilingShape(Protocol):
    """Protocol for tiling shapes used in geometric partitioning."""

    name: str

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return translation vectors for the tiling lattice."""
        ...

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return True if a Cartesian point lies inside the cell centered at origin."""
        ...

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return candidate cell centers within bounds."""
        ...


TILING_REGISTRY: dict[str, TilingShape] = {}


def register_tiling_shape(name: str, shape: TilingShape) -> None:
    """Register a tiling shape with the global registry."""
    key = name.strip().lower()
    if key in TILING_REGISTRY:
        raise ValueError(f"Tiling shape '{key}' is already registered.")
    TILING_REGISTRY[key] = shape


def list_tiling_shapes() -> list[str]:
    """List registered tiling shape names."""
    return sorted(TILING_REGISTRY.keys())


def get_tiling_shape(name: str) -> TilingShape:
    """Retrieve a tiling shape by name with helpful errors."""
    key = name.strip().lower()
    if key not in TILING_REGISTRY:
        available = ", ".join(list_tiling_shapes())
        raise KeyError(f"Unknown tiling shape '{name}'. Available: {available}")
    return TILING_REGISTRY[key]


def grid_shape_from_count(n_cells: int) -> tuple[int, int, int]:
    """Find a (nx, ny, nz) grid shape whose product equals n_cells."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    best = (n_cells, 1, 1)
    best_ratio = float("inf")
    for nx in range(1, n_cells + 1):
        if n_cells % nx != 0:
            continue
        rem = n_cells // nx
        for ny in range(1, rem + 1):
            if rem % ny != 0:
                continue
            nz = rem // ny
            dims = sorted([nx, ny, nz])
            ratio = dims[-1] / dims[0]
            if ratio < best_ratio:
                best_ratio = ratio
                best = (nx, ny, nz)
    return best


def _cell_centers_axis_aligned(
    n_cells: int,
    bounds: np.ndarray,
) -> np.ndarray:
    """Generate cell centers on an axis-aligned grid."""
    grid_shape = grid_shape_from_count(n_cells)
    mins = bounds[0]
    maxs = bounds[1]
    centers_per_axis = []
    for axis, count in enumerate(grid_shape):
        edges = np.linspace(mins[axis], maxs[axis], num=count + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        centers_per_axis.append(centers)
    gx, gy, gz = np.meshgrid(*centers_per_axis, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _hexagon_contains(x: float, y: float, side_length: float) -> bool:
    """Check if a point lies inside a regular hexagon (flat top)."""
    ax = abs(x)
    ay = abs(y)
    limit_y = np.sqrt(3) * side_length / 2.0
    return (
        ax <= side_length
        and ay <= limit_y
        and (ax + ay / np.sqrt(3)) <= side_length
    )


class CubeTiling:
    """Cube tiling with axis-aligned lattice vectors."""

    name = "cube"

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return or compute lattice vectors."""
        return np.eye(3) * scale

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return or compute point in cell."""
        half = scale / 2.0
        return bool(np.all(np.abs(point) <= half + 1e-12))

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return or compute cell centers."""
        return _cell_centers_axis_aligned(n_cells, bounds)


class HexagonalPrismTiling:
    """Hexagonal prism tiling aligned with the z-axis."""

    name = "hex_prism"

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return or compute lattice vectors."""
        a1 = np.array([scale, 0.0, 0.0])
        a2 = np.array([scale / 2.0, np.sqrt(3) * scale / 2.0, 0.0])
        c = np.array([0.0, 0.0, scale])
        return np.vstack([a1, a2, c])

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return or compute point in cell."""
        x, y, z = point
        side = scale / np.sqrt(3.0)
        return _hexagon_contains(x, y, side) and abs(z) <= scale / 2.0 + 1e-12

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return or compute cell centers."""
        return _cell_centers_axis_aligned(n_cells, bounds)


class TruncatedOctahedronTiling:
    """Truncated octahedron tiling based on BCC Wigner-Seitz cells."""

    name = "truncated_octahedron"

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return or compute lattice vectors."""
        a = scale / 2.0
        return np.array(
            [
                [a, a, -a],
                [-a, a, a],
                [a, -a, a],
            ]
        )

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return or compute point in cell."""
        x, y, z = point
        half = scale / 2.0
        if max(abs(x), abs(y), abs(z)) > half + 1e-12:
            return False
        return (abs(x) + abs(y) + abs(z)) <= (3.0 * scale / 4.0 + 1e-12)

    def vertex_set(self, scale: float) -> np.ndarray:
        """Return or compute vertex set."""
        a = scale
        verts = []
        coords = [(-a / 2, 0, -a / 4), (-a / 2, 0, a / 4), (a / 2, 0, -a / 4), (a / 2, 0, a / 4)]
        for x, y, z in coords:
            verts.append([x, y, z])
            verts.append([y, x, z])
            verts.append([x, z, y])
        return np.unique(np.array(verts), axis=0)

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return or compute cell centers."""
        return _cell_centers_axis_aligned(n_cells, bounds)


class RhombicDodecahedronTiling:
    """Rhombic dodecahedron tiling based on FCC Wigner-Seitz cells."""

    name = "rhombic_dodecahedron"

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return or compute lattice vectors."""
        a = scale / 2.0
        return np.array(
            [
                [0.0, a, a],
                [a, 0.0, a],
                [a, a, 0.0],
            ]
        )

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return or compute point in cell."""
        x, y, z = point
        half = scale / 2.0
        return (
            (abs(x) + abs(y) <= half + 1e-12)
            and (abs(x) + abs(z) <= half + 1e-12)
            and (abs(y) + abs(z) <= half + 1e-12)
        )

    def vertex_set(self, scale: float) -> np.ndarray:
        """Return or compute vertex set."""
        half = scale / 2.0
        quarter = scale / 4.0
        verts = [
            [half, 0.0, 0.0],
            [-half, 0.0, 0.0],
            [0.0, half, 0.0],
            [0.0, -half, 0.0],
            [0.0, 0.0, half],
            [0.0, 0.0, -half],
        ]
        for sx in (-quarter, quarter):
            for sy in (-quarter, quarter):
                for sz in (-quarter, quarter):
                    verts.append([sx, sy, sz])
        return np.array(verts)

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return or compute cell centers."""
        return _cell_centers_axis_aligned(n_cells, bounds)


class ElongatedDodecahedronTiling:
    """Elongated dodecahedron tiling as stretched rhombic dodecahedron."""

    name = "elongated_dodecahedron"

    def __init__(self, elongation: float = 1.5):
        """Initialize a new ElongatedDodecahedronTiling instance."""
        if elongation <= 0:
            raise ValueError("elongation must be positive.")
        self.elongation = elongation

    def lattice_vectors(self, scale: float) -> np.ndarray:
        """Return or compute lattice vectors."""
        base = RhombicDodecahedronTiling().lattice_vectors(scale)
        stretched = base.copy()
        stretched[:, 2] *= self.elongation
        return stretched

    def point_in_cell(self, point: np.ndarray, scale: float) -> bool:
        """Return or compute point in cell."""
        scaled = np.array([point[0], point[1], point[2] / self.elongation])
        return RhombicDodecahedronTiling().point_in_cell(scaled, scale)

    def cell_centers(self, n_cells: int, bounds: np.ndarray) -> np.ndarray:
        """Return or compute cell centers."""
        return _cell_centers_axis_aligned(n_cells, bounds)


def _register_defaults() -> None:
    """Internal helper to register defaults."""
    register_tiling_shape(CubeTiling.name, CubeTiling())
    register_tiling_shape(HexagonalPrismTiling.name, HexagonalPrismTiling())
    register_tiling_shape(TruncatedOctahedronTiling.name, TruncatedOctahedronTiling())
    register_tiling_shape(ElongatedDodecahedronTiling.name, ElongatedDodecahedronTiling())
    register_tiling_shape(RhombicDodecahedronTiling.name, RhombicDodecahedronTiling())


_register_defaults()
