# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Hungarian algorithm for optimal assignment.

This module implements the Hungarian (Munkres) algorithm for solving
the minimum cost assignment problem, used for matching molecules
between reference and target structures.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hungarian_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the minimum cost assignment problem using the Hungarian algorithm.

    Given a cost matrix, find the optimal assignment of rows to columns
    that minimizes the total cost.

    Parameters
    ----------
    cost : np.ndarray
        Cost matrix with shape (N, N). cost[i, j] is the cost of
        assigning row i to column j.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (row_indices, col_indices) such that row_indices[k] is assigned
        to col_indices[k] in the optimal solution.

    Notes
    -----
    This is a pure NumPy implementation of the Hungarian algorithm.
    For large matrices, scipy.optimize.linear_sum_assignment may be faster.
    """
    cost = np.array(cost, dtype=float)
    n = cost.shape[0]
    tol = 1e-12

    # Step 1: Reduce rows and columns
    cost -= cost.min(axis=1)[:, None]
    cost -= cost.min(axis=0)[None, :]

    # Initialize starred and primed zeros
    star = np.zeros((n, n), dtype=bool)
    prime = np.zeros((n, n), dtype=bool)
    row_cov = np.zeros(n, dtype=bool)
    col_cov = np.zeros(n, dtype=bool)

    def is_zero(x: float) -> bool:
        """Return or compute is zero."""
        return bool(np.isclose(x, 0.0, atol=tol))

    # Star zeros (one per row/column)
    for i in range(n):
        for j in range(n):
            if is_zero(cost[i, j]) and not row_cov[i] and not col_cov[j]:
                star[i, j] = True
                row_cov[i] = True
                col_cov[j] = True

    row_cov[:] = False
    col_cov[:] = False

    def cover_starred_columns() -> None:
        """Return or compute cover starred columns."""
        for j in range(n):
            col_cov[j] = bool(star[:, j].any())

    def find_uncovered_zero() -> Tuple[int, int] | None:
        """Return or compute find uncovered zero."""
        z = np.isclose(cost, 0.0, atol=tol) & (~row_cov[:, None]) & (~col_cov[None, :])
        where = np.argwhere(z)
        if where.size == 0:
            return None
        return int(where[0][0]), int(where[0][1])

    def min_uncovered_value() -> float:
        """Return or compute min uncovered value."""
        mask = (~row_cov[:, None]) & (~col_cov[None, :])
        vals = cost[mask]
        return float(vals.min()) if vals.size else 0.0

    cover_starred_columns()

    while int(col_cov.sum()) < n:
        z = find_uncovered_zero()
        if z is None:
            # Step 6: Adjust the matrix
            m = min_uncovered_value()
            cost[row_cov, :] += m
            cost[:, ~col_cov] -= m
            continue

        i, j = z
        prime[i, j] = True

        # Check for starred zero in this row
        star_cols = np.where(star[i, :])[0]
        if star_cols.size > 0:
            # Cover this row, uncover the starred column
            row_cov[i] = True
            col_cov[int(star_cols[0])] = False
            continue

        # Step 5: Augmenting path
        path = [(i, j)]
        cur_j = j
        while True:
            star_rows = np.where(star[:, cur_j])[0]
            if star_rows.size == 0:
                break
            r = int(star_rows[0])
            path.append((r, cur_j))
            prime_cols = np.where(prime[r, :])[0]
            c = int(prime_cols[0])
            path.append((r, c))
            cur_j = c

        # Augment: unstar starred, star primed
        for r, c in path:
            star[r, c] = not star[r, c]

        # Clear primes and covers
        prime[:, :] = False
        row_cov[:] = False
        col_cov[:] = False
        cover_starred_columns()

    # Extract the solution
    rows = np.arange(n, dtype=int)
    cols = np.full(n, -1, dtype=int)
    for i in range(n):
        js = np.where(star[i, :])[0]
        cols[i] = int(js[0])

    return rows, cols
