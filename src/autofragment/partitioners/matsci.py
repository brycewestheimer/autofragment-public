# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Materials science partitioner."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple

from autofragment.core.types import (
    ChemicalSystem,
    Fragment,
    FragmentTree,
)
from autofragment.partitioners.base import BasePartitioner

if TYPE_CHECKING:
    pass

class MatSciPartitioner(BasePartitioner):
    """Base partitioner for materials/periodic systems.

    Handles:
    - Periodic boundary conditions
    - Lattice-aware operations
    - Surface vs bulk distinction
    """

    def __init__(
        self,
        use_supercell: bool = False,
        supercell_size: Tuple[int, int, int] = (1, 1, 1),
        treat_as_molecular: bool = False
    ):
        """Initialize materials partitioner.

        Args:
            use_supercell: Expand to supercell before fragmenting
            supercell_size: Repetitions along each axis
            treat_as_molecular: Treat as finite, not periodic
        """
        super().__init__()
        self.use_supercell = use_supercell
        self.supercell_size = supercell_size
        self.treat_as_molecular = treat_as_molecular

    def partition(
        self,
        system: ChemicalSystem,
        source_file: str | None = None,
    ) -> FragmentTree:
        """Partition materials system."""
        if not system.periodic:
            self.treat_as_molecular = True

        # Optionally expand to supercell
        if self.use_supercell and system.periodic:
            from autofragment.core.periodic import make_supercell
            system = make_supercell(system, self.supercell_size)

        # Call implementation-specific method
        fragments = self._partition_impl(system)

        # Add periodic metadata
        for frag in fragments:
            frag.metadata["periodic_origin"] = system.periodic
            if system.periodic and system.lattice:
                 # Should we store lattice info in fragment?
                 pass

        return FragmentTree(
            fragments=fragments,
            source={"file": source_file} if source_file else {},
            partitioning={"method": self.__class__.__name__}
        )

    @abstractmethod
    def _partition_impl(
        self,
        system: "ChemicalSystem"
    ) -> List["Fragment"]:
        """Implementation-specific partitioning."""
        pass
