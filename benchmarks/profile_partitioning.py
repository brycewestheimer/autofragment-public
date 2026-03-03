# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Profiling script for autofragment."""
from __future__ import annotations

import cProfile
from pathlib import Path
import pstats

from autofragment.algorithms.graph_partition import min_cut_partition
from autofragment.io.readers import read_pdb


def profile_partitioning() -> pstats.Stats:
    """Profile fragmentation of a large system."""
    path = Path("benchmarks/data/large_protein.pdb")
    if not path.exists():
        raise FileNotFoundError(
            "Benchmark PDB not found. Place a large PDB at "
            f"{path} to run profiling."
        )

    system = read_pdb(path)
    graph = system.to_graph()

    profiler = cProfile.Profile()
    profiler.enable()

    _ = min_cut_partition(graph, n_partitions=4)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    return stats


if __name__ == "__main__":
    profile_partitioning()
