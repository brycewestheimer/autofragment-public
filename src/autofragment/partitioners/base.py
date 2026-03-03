# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for partitioners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from autofragment.core.types import ChemicalSystem, FragmentTree


@dataclass(frozen=True)
class TilingSelection:
    """Configuration for tiling-based partitioners."""

    tiling_shape: str = "cube"
    n_fragments: int = 1
    tiling_scale: float = 1.0


def validate_tiling_shape(name: str) -> str:
    """Validate tiling shape name and return normalized key."""
    from autofragment.geometry.tilings import get_tiling_shape

    shape = get_tiling_shape(name)
    return shape.name


class BasePartitioner(ABC):
    """
    Abstract base class for molecular partitioners.

    Subclasses must implement the `partition` method that takes a
    ChemicalSystem and returns a FragmentTree.
    """

    @abstractmethod
    def partition(
        self,
        system: ChemicalSystem,
        source_file: str | None = None,
    ) -> FragmentTree:
        """
        Partition a chemical system into fragments.

        Parameters
        ----------
        system : ChemicalSystem
            Chemical system to partition.
        source_file : str, optional
            Path to the source file (for metadata).

        Returns
        -------
        FragmentTree
            Fragmentation result containing a flat fragments list.
        """
        pass
