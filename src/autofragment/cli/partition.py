# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""CLI helpers for tiling-based partitioning."""
from __future__ import annotations

import argparse

from autofragment.geometry.tilings import list_tiling_shapes


def add_tiling_options(parser: argparse.ArgumentParser) -> None:
    """Add tiling options to an argparse parser."""
    shapes = list_tiling_shapes()
    parser.add_argument(
        "--tiling-shape",
        choices=shapes,
        default="cube",
        help=f"Tiling shape ({', '.join(shapes)})",
    )
    parser.add_argument(
        "--n-fragments",
        type=int,
        default=8,
        help="Target number of fragments (default: 8)",
    )
    parser.add_argument(
        "--tiling-scale",
        type=float,
        default=1.0,
        help="Tiling scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--bounds-strategy",
        choices=["auto", "lattice", "atoms"],
        default="auto",
        help="Bounding strategy for tiling (default: auto)",
    )
