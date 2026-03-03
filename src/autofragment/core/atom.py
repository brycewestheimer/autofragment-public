# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Memory-efficient atom storage utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

atom_dtype = np.dtype(
    [
        ("element", "U2"),
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("formal_charge", "i1"),
        ("atom_type", "i2"),
    ]
)


@dataclass
class CompactAtomArray:
    """Memory-efficient atom storage using structured numpy arrays."""

    n_atoms: int

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        self._data = np.zeros(self.n_atoms, dtype=atom_dtype)

    def __len__(self) -> int:
        """Return the number of contained items."""
        return len(self._data)

    @property
    def positions(self) -> np.ndarray:
        """Return positions as (N, 3) float array."""
        return np.column_stack([
            self._data["x"],
            self._data["y"],
            self._data["z"],
        ])

    def set_position(self, index: int, coords: np.ndarray) -> None:
        """Set position for an atom index."""
        self._data[index]["x"] = coords[0]
        self._data[index]["y"] = coords[1]
        self._data[index]["z"] = coords[2]

    def set_element(self, index: int, symbol: str) -> None:
        """Set element symbol for an atom index."""
        self._data[index]["element"] = symbol

    @property
    def data(self) -> np.ndarray:
        """Return underlying structured array."""
        return self._data
