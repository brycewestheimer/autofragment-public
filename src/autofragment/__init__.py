# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""autofragment - Molecular fragmentation for computational chemistry.

This package provides tools for partitioning molecular systems into a flat list
of fragments suitable for many-body expansion calculations.

Quick Start
-----------
>>> import autofragment as af
>>> result = af.partition_xyz("water64.xyz", n_fragments=4)
>>> result.to_json("partitioned.json")

For full control:
>>> partitioner = af.MolecularPartitioner(n_fragments=4)
>>> system = af.io.read_xyz("water64.xyz")
>>> result = partitioner.partition(system)
"""

from autofragment._version import __version__
from autofragment.algorithms.seeding import SEEDING_STRATEGIES, compute_seeds
from autofragment.core.bonds import InterfragmentBond
from autofragment.core.graph import MolecularGraph
from autofragment.core.types import (
    Atom,
    ChemicalSystem,
    Fragment,
    FragmentationScheme,
    FragmentTree,
    Molecule,
    molecules_to_system,
    system_to_molecules,
)
from autofragment.features import has_feature, list_features
from autofragment.partitioners.batch import BatchPartitioner
from autofragment.partitioners.bio import BioPartitioner
from autofragment.partitioners.geometric import (
    RadialPartitioner,
    SlabPartitioner,
    SurfacePartitioner,
    TilingPartitioner,
    UnitCellPartitioner,
)
from autofragment.partitioners.molecular import MolecularPartitioner
from autofragment.partitioners.spectral import SpectralPartitioner


# Convenience function for quick partitioning
def partition_xyz(
    filepath: str,
    *,
    n_fragments: int = 4,
    method: str = "kmeans",
    random_state: int = 42,
    tiers: int | None = None,
    n_primary: int | None = None,
    n_secondary: int | None = None,
    n_tertiary: int | None = None,
    init_strategy: str | None = None,
) -> FragmentTree:
    """
    Partition a water cluster XYZ file into fragments.

    This is a convenience function that combines reading the XYZ file
    and partitioning it in a single call. Supports both flat and tiered
    partitioning.

    Parameters
    ----------
    filepath : str
        Path to the XYZ file containing water molecules.
    n_fragments : int, optional
        Number of fragments (flat mode). Default is 4.
    method : str, optional
        Clustering method. Default is "kmeans".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    tiers : int, optional
        Number of hierarchy tiers (2 or 3). None = flat mode.
    n_primary : int, optional
        Number of primary fragments (tiered mode).
    n_secondary : int, optional
        Number of secondary fragments per primary (tiered mode).
    n_tertiary : int, optional
        Number of tertiary fragments per secondary (3-tier mode).
    init_strategy : str, optional
        K-means seeding strategy (e.g. "pca", "axis", "halfplane", "radial").

    Returns
    -------
    FragmentTree
        Fragmentation result.

    Examples
    --------
    Flat mode:

    >>> result = af.partition_xyz("water64.xyz", n_fragments=4)
    >>> result.to_json("output.json")

    Tiered mode:

    >>> result = af.partition_xyz(
    ...     "water64.xyz", tiers=2, n_primary=4, n_secondary=4
    ... )
    """
    from autofragment import io

    system = io.read_xyz(filepath)

    if tiers is not None:
        partitioner = MolecularPartitioner(
            tiers=tiers,
            n_primary=n_primary,
            n_secondary=n_secondary,
            n_tertiary=n_tertiary,
            method=method,
            random_state=random_state,
            init_strategy=init_strategy,
        )
    else:
        partitioner = MolecularPartitioner(
            n_fragments=n_fragments,
            method=method,
            random_state=random_state,
            init_strategy=init_strategy,
        )
    return partitioner.partition(system, source_file=filepath)


__all__ = [
    "__version__",
    "Atom",
    "Molecule",
    "Fragment",
    "FragmentTree",
    "InterfragmentBond",
    "ChemicalSystem",
    "FragmentationScheme",
    "molecules_to_system",
    "system_to_molecules",
    "MolecularGraph",
    "MolecularPartitioner",
    "BatchPartitioner",
    "RadialPartitioner",
    "SlabPartitioner",
    "UnitCellPartitioner",
    "SurfacePartitioner",
    "TilingPartitioner",
    "BioPartitioner",
    "SpectralPartitioner",
    "compute_seeds",
    "SEEDING_STRATEGIES",
    "has_feature",
    "list_features",
    "partition_xyz",
]
