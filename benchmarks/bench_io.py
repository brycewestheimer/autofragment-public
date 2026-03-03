# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""I/O benchmarks for reading XYZ files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autofragment import io


def _write_xyz(path: Path, n_molecules: int) -> None:
    n_atoms = n_molecules * 3
    lines = [str(n_atoms), "Generated water cluster"]
    for i in range(n_molecules):
        offset = np.array([i * 3.0, 0.0, 0.0])
        lines.append(f"O {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}")
        lines.append(f"H {offset[0] + 0.96:.3f} {offset[1]:.3f} {offset[2]:.3f}")
        lines.append(f"H {offset[0] - 0.24:.3f} {offset[1] + 0.93:.3f} {offset[2]:.3f}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def xyz_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "cluster.xyz"
    _write_xyz(path, n_molecules=500)
    return path


def test_read_xyz(benchmark, xyz_file):
    system = benchmark(io.read_xyz, xyz_file)
    assert system.n_atoms > 0
