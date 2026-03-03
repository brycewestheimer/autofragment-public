# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel processing utilities."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 0,
) -> List[R]:
    """Apply function to items in parallel.

    Parameters
    ----------
    func : Callable
        Function to apply to each item
    items : Sequence
        Items to process
    n_jobs : int, optional
        Number of parallel jobs (-1 for all CPUs)
    backend : str
        Joblib backend ("loky", "threading", "multiprocessing")
    verbose : int
        Verbosity level

    Returns
    -------
    list
        Results in same order as input
    """
    try:
        from joblib import Parallel, delayed  # type: ignore[import-untyped]
    except ImportError:
        return [func(item) for item in items]

    if n_jobs is None:
        n_jobs = -1

    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item) for item in items
    )


def parallel_partition_many(partitioner, systems: Sequence, n_jobs: int = -1) -> List:
    """Partition multiple systems in parallel."""
    return parallel_map(partitioner.partition, systems, n_jobs=n_jobs)
