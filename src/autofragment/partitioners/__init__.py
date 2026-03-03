# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Partitioner classes for molecular fragmentation."""

from autofragment.partitioners.base import BasePartitioner
from autofragment.partitioners.batch import BatchPartitioner
from autofragment.partitioners.bio import BioPartitioner
from autofragment.partitioners.geometric import TilingPartitioner
from autofragment.partitioners.molecular import MolecularPartitioner
from autofragment.partitioners.nucleic import NucleicPartitioner
from autofragment.partitioners.qmmm import (
    AtomSelection,
    CombinedSelection,
    DistanceSelection,
    QMMMPartitioner,
    QMMMResult,
    QMSelection,
    ResidueSelection,
    TopologySelection,
)
from autofragment.partitioners.spectral import SpectralPartitioner
from autofragment.partitioners.topology import ShellSelectionResult, TopologyNeighborSelection

__all__ = [
    "BasePartitioner",
    "MolecularPartitioner",
    "BatchPartitioner",
    "BioPartitioner",
    "NucleicPartitioner",
    "TilingPartitioner",
    "SpectralPartitioner",
    # QM/MM
    "QMMMPartitioner",
    "QMMMResult",
    "QMSelection",
    "AtomSelection",
    "ResidueSelection",
    "DistanceSelection",
    "CombinedSelection",
    "TopologySelection",
    # Shared topology neighborhood selection
    "TopologyNeighborSelection",
    "ShellSelectionResult",
]
