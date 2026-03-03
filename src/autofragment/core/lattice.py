# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Lattice vector representation and operations."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Lattice:
    """Crystallographic lattice representation.

    Stores lattice vectors and provides common operations.
    """
    vectors: np.ndarray  # 3x3 matrix, rows are a, b, c vectors

    @classmethod
    def from_parameters(
        cls,
        a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float
    ) -> "Lattice":
        """Create lattice from cell parameters.

        Args:
            a, b, c: Lattice vector lengths (Angstrom)
            alpha, beta, gamma: Angles (degrees)

        Returns:
            Lattice object
        """
        # Convert angles to radians
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)

        # Build matrix using crystallographic convention
        cos_alpha = np.cos(alpha_r)
        cos_beta = np.cos(beta_r)
        cos_gamma = np.cos(gamma_r)
        sin_gamma = np.sin(gamma_r)

        # Volume factor
        val = (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        vectors = np.array([
            [a, 0, 0],
            [b * cos_gamma, b * sin_gamma, 0],
            [c * cos_beta, c * val, c * np.sqrt(1 - cos_beta**2 - val**2)]
        ])

        return cls(vectors=vectors)

    @property
    def volume(self) -> float:
        """Cell volume in cubic Angstrom."""
        return abs(np.linalg.det(self.vectors))

    @property
    def reciprocal(self) -> "Lattice":
        """Get reciprocal lattice."""
        recip = 2 * np.pi * np.linalg.inv(self.vectors).T
        return Lattice(vectors=recip)

    def fractional_to_cartesian(self, frac: np.ndarray) -> np.ndarray:
        """Convert fractional to Cartesian coordinates."""
        return frac @ self.vectors

    def cartesian_to_fractional(self, cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian to fractional coordinates."""
        return cart @ np.linalg.inv(self.vectors)
