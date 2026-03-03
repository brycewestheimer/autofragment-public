# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry utilities for tiling partitioning."""

from autofragment.geometry.tilings import (
    TilingCell,
    TilingShape,
    get_tiling_shape,
    grid_shape_from_count,
    list_tiling_shapes,
    register_tiling_shape,
)

__all__ = [
    "TilingCell",
    "TilingShape",
    "get_tiling_shape",
    "list_tiling_shapes",
    "register_tiling_shape",
    "grid_shape_from_count",
]
