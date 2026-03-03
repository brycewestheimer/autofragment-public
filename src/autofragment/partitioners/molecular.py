# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Molecular partitioner for water clusters and similar systems.

This module provides the main partitioner class for creating fragments
from collections of molecules (typically water clusters). Supports both
flat partitioning and tiered hierarchical partitioning (2-tier and 3-tier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autofragment.algorithms.clustering import partition_labels
from autofragment.algorithms.geometric import partition_by_planes, partition_by_planes_tiered
from autofragment.core.geometry import compute_centroids
from autofragment.core.types import (
    ChemicalSystem,
    Fragment,
    FragmentTree,
    Molecule,
    system_to_molecules,
)
from autofragment.io.output import format_partitioning_info, format_source_info
from autofragment.partitioners.base import BasePartitioner
from autofragment.partitioners.topology import (
    BondPolicy,
    SelectionMode,
    TopologyNeighborSelection,
)

# Type aliases for tiered labels
LabelTuple2 = Tuple[int, int]
LabelTuple3 = Tuple[int, int, int]


class PartitionError(ValueError):
    """Raised when partitioning fails."""

    pass


@dataclass(frozen=True)
class PartitionResult:
    """Result of flat partitioning with labels and metadata."""

    n_fragments: int
    labels: np.ndarray
    centroids: np.ndarray
    molecules: List[Molecule]


@dataclass(frozen=True)
class TieredPartitionResult:
    """Result of tiered partitioning with tuple labels and metadata."""

    tiers: int
    n_primary: int
    n_secondary: int
    n_tertiary: Optional[int]
    labels: List[Union[LabelTuple2, LabelTuple3]]
    centroids: np.ndarray
    molecules: List[Molecule]


class MolecularPartitioner(BasePartitioner):
    """
    Partitioner for water clusters and similar molecular systems.

    Supports both flat partitioning (default) and tiered hierarchical
    partitioning (2-tier and 3-tier).

    Parameters
    ----------
    n_fragments : int
        Number of fragments (flat mode). Default is 4.
    method : str, optional
        Clustering method. Default is "kmeans".
    random_state : int, optional
        Random seed for clustering. Default is 42.
    strict_balanced : bool, optional
        If True, validate equal cluster sizes. Default is True for
        kmeans_constrained, False otherwise.
    tiers : int, optional
        Number of hierarchy tiers (2 or 3). None = flat mode (default).
    n_primary : int, optional
        Number of primary fragments (tiered mode).
    n_secondary : int, optional
        Number of secondary fragments per primary (tiered mode).
    n_tertiary : int, optional
        Number of tertiary fragments per secondary (3-tier mode).
    init_strategy : str | ndarray | dict | None, optional
        Default seeding strategy for all tiers / flat mode.
    init_strategy_primary : str | ndarray | dict | None, optional
        Override seeding strategy for primary (tier-1) clustering.
    init_strategy_secondary : str | ndarray | dict | None, optional
        Override seeding strategy for secondary (tier-2) clustering.
    init_strategy_tertiary : str | ndarray | dict | None, optional
        Override seeding strategy for tertiary (tier-3) clustering.

    Examples
    --------
    Flat mode:

    >>> partitioner = MolecularPartitioner(n_fragments=2, method="kmeans")
    >>> tree = partitioner.partition(system)
    >>> len(tree.fragments)
    2

    Tiered mode:

    >>> partitioner = MolecularPartitioner(
    ...     tiers=2, n_primary=4, n_secondary=4
    ... )
    >>> tree = partitioner.partition(system)
    >>> tree.n_primary
    4

    See Also
    --------
    autofragment.partitioners.batch.BatchPartitioner : For processing multiple files.
    autofragment.core.types.FragmentTree : The result object containing fragments.
    """

    def __init__(
        self,
        n_fragments: int = 4,
        method: str = "kmeans",
        random_state: int = 42,
        strict_balanced: Optional[bool] = None,
        topology_refine: bool = False,
        topology_mode: SelectionMode = "graph",
        topology_hops: int = 1,
        topology_layers: int = 1,
        topology_k_per_layer: int = 1,
        topology_bond_policy: BondPolicy = "infer",
        # Tiered parameters
        tiers: Optional[int] = None,
        n_primary: Optional[int] = None,
        n_secondary: Optional[int] = None,
        n_tertiary: Optional[int] = None,
        # Seeding parameters
        init_strategy: Union[None, str, np.ndarray, Dict[str, Any]] = None,
        init_strategy_primary: Union[None, str, np.ndarray, Dict[str, Any]] = None,
        init_strategy_secondary: Union[None, str, np.ndarray, Dict[str, Any]] = None,
        init_strategy_tertiary: Union[None, str, np.ndarray, Dict[str, Any]] = None,
    ):
        """Initialize a new MolecularPartitioner instance."""
        self.tiers = tiers
        self.method = method
        self.random_state = random_state
        self.topology_refine = topology_refine
        self.topology_mode = topology_mode
        self.topology_hops = topology_hops
        self.topology_layers = topology_layers
        self.topology_k_per_layer = topology_k_per_layer
        self.topology_bond_policy = topology_bond_policy

        self.n_primary: Optional[int] = None
        self.n_secondary: Optional[int] = None
        self.n_tertiary: Optional[int] = None

        if tiers is not None:
            # Tiered mode
            if tiers not in (2, 3):
                raise ValueError(f"tiers must be 2 or 3, got {tiers}")
            if n_primary is None or n_secondary is None:
                raise ValueError("n_primary and n_secondary are required for tiered mode")
            if tiers == 3 and n_tertiary is None:
                raise ValueError("n_tertiary is required for 3-tier partitioning")
            self.n_primary = n_primary
            self.n_secondary = n_secondary
            self.n_tertiary = n_tertiary
            # n_fragments is still set for compatibility (total leaf fragments)
            total = n_primary * n_secondary
            if tiers == 3 and n_tertiary is not None:
                total *= n_tertiary
            self.n_fragments = total
        else:
            # Flat mode
            if n_fragments <= 0:
                raise ValueError(f"n_fragments must be positive, got {n_fragments}")
            self.n_fragments = n_fragments

        # Per-tier init strategies: per-tier override > general > None
        self._init_strategy = init_strategy
        self._init_primary = (
            init_strategy_primary if init_strategy_primary is not None else init_strategy
        )
        self._init_secondary = (
            init_strategy_secondary if init_strategy_secondary is not None else init_strategy
        )
        self._init_tertiary = (
            init_strategy_tertiary if init_strategy_tertiary is not None else init_strategy
        )

        if strict_balanced is None:
            self.strict_balanced = method == "kmeans_constrained"
        else:
            self.strict_balanced = strict_balanced

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
            Path to source file for metadata.

        Returns
        -------
        FragmentTree
            Fragmentation result. Flat for non-tiered mode, hierarchical
            for tiered mode.
        """
        molecules = system_to_molecules(system, require_metadata=True)

        if self.tiers is not None:
            return self._partition_tiered(list(molecules), source_file)

        # --- Flat mode (existing path) ---
        result = self._build_partition(list(molecules))

        if self.topology_refine:
            refined_labels = self._refine_partition_topology(system, result)
            result = PartitionResult(
                n_fragments=result.n_fragments,
                labels=refined_labels,
                centroids=result.centroids,
                molecules=result.molecules,
            )

        if self.strict_balanced:
            self._validate_partition(result)

        fragments = self._build_fragments(result)

        # Build metadata
        source = {}
        if source_file:
            source = format_source_info(source_file, "xyz")

        partitioning = format_partitioning_info(
            algorithm=self.method,
            n_fragments=self.n_fragments,
        )

        return FragmentTree(
            fragments=fragments,
            source=source,
            partitioning=partitioning,
        )

    def _build_partition(self, molecules: List[Molecule]) -> PartitionResult:
        """Build the partition with labels (flat mode)."""
        centroids = compute_centroids(molecules)

        if self.method == "geom_planes":
            labels = partition_by_planes(centroids, self.n_fragments)
        else:
            labels = partition_labels(
                centroids, self.n_fragments, self.method, self.random_state,
                init=self._init_strategy,
            )

        return PartitionResult(
            n_fragments=self.n_fragments,
            labels=labels.astype(int),
            centroids=centroids,
            molecules=molecules,
        )

    def _molecule_atom_indices(
        self,
        system: ChemicalSystem,
        molecules: List[Molecule],
    ) -> List[List[int]]:
        """Get atom indices for each molecule in order."""
        metadata = system.metadata or {}
        if "molecule_atom_indices" in metadata:
            return [list(indices) for indices in metadata["molecule_atom_indices"]]

        indices: List[List[int]] = []
        start = 0
        for mol in molecules:
            end = start + len(mol)
            indices.append(list(range(start, end)))
            start = end
        return indices

    def _refine_partition_topology(
        self,
        system: ChemicalSystem,
        result: PartitionResult,
    ) -> np.ndarray:
        """Optionally refine molecule labels using topology neighborhoods around cluster centers."""
        labels = result.labels.copy().astype(int)
        n_molecules = len(result.molecules)
        if n_molecules == 0:
            return labels

        molecule_indices = self._molecule_atom_indices(system, result.molecules)
        coords = np.array([atom.coords for atom in system.atoms], dtype=float)
        elements = [atom.symbol for atom in system.atoms]
        bonds = [
            (int(bond["atom1"]), int(bond["atom2"]))
            for bond in system.bonds
            if "atom1" in bond and "atom2" in bond
        ]

        # Representative molecule per cluster = one closest to cluster centroid.
        representatives: Dict[int, int] = {}
        for cluster_idx in range(self.n_fragments):
            members = np.where(labels == cluster_idx)[0].tolist()
            if not members:
                continue
            cluster_center = np.mean(result.centroids[members], axis=0)
            rep = min(
                members,
                key=lambda idx: float(np.linalg.norm(result.centroids[idx] - cluster_center)),
            )
            representatives[cluster_idx] = rep

        if not representatives:
            return labels

        overlap_scores = np.zeros((self.n_fragments, n_molecules), dtype=int)
        for cluster_idx, rep_idx in representatives.items():
            selector = TopologyNeighborSelection(
                seed_atoms=set(molecule_indices[rep_idx]),
                mode=self.topology_mode,
                hops=self.topology_hops,
                layers=self.topology_layers,
                k_per_layer=self.topology_k_per_layer,
                expand_residues=False,
                bond_policy=self.topology_bond_policy,
            )
            selected_atoms = selector.select(coords, elements, bonds=bonds).selected_atoms

            for mol_idx, atom_ids in enumerate(molecule_indices):
                overlap_scores[cluster_idx, mol_idx] = len(selected_atoms.intersection(atom_ids))

        refined = labels.copy()
        for mol_idx in range(n_molecules):
            current = int(labels[mol_idx])
            best_cluster = current
            best_overlap = int(overlap_scores[current, mol_idx])
            for cluster_idx in range(self.n_fragments):
                overlap = int(overlap_scores[cluster_idx, mol_idx])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = cluster_idx
            if best_overlap > 0:
                refined[mol_idx] = best_cluster

        # Preserve valid non-empty partitioning; otherwise keep original labels.
        counts = [int(np.sum(refined == k)) for k in range(self.n_fragments)]
        if any(count == 0 for count in counts):
            return labels
        return refined

    def _validate_partition(self, result: PartitionResult) -> None:
        """Validate that cluster sizes are equal (when requested)."""
        n_molecules = len(result.molecules)
        if n_molecules % self.n_fragments != 0:
            raise PartitionError(
                f"Non-integer cluster sizes: {n_molecules} molecules / {self.n_fragments} fragments"
            )

        expected = n_molecules // self.n_fragments
        for k in range(self.n_fragments):
            count_k = int(np.sum(result.labels == k))
            if count_k != expected:
                raise PartitionError(
                    f"Fragment {k}: size {count_k} != expected {expected}"
                )

    def _build_fragments(self, result: PartitionResult) -> List[Fragment]:
        """Build Fragment objects from partition result (flat mode)."""
        fragments: List[Fragment] = []

        for k in range(self.n_fragments):
            chosen = [
                result.molecules[i]
                for i, lbl in enumerate(result.labels.tolist())
                if int(lbl) == k
            ]
            if not chosen:
                raise PartitionError(f"Fragment {k} is empty")
            f = Fragment.from_molecules(chosen, f"F{k + 1}")
            f.metadata["n_molecules"] = len(chosen)
            fragments.append(f)

        return fragments

    # ------------------------------------------------------------------
    # Tiered partitioning
    # ------------------------------------------------------------------

    def _partition_tiered(
        self,
        molecules: List[Molecule],
        source_file: str | None = None,
    ) -> FragmentTree:
        """Orchestrate hierarchical clustering and build a tiered FragmentTree."""
        result = self._build_partition_tiered(molecules)

        if self.strict_balanced:
            self._validate_tiered_hierarchy(result)

        fragments = self._build_tiered_fragments(result)

        # Build metadata
        source = {}
        if source_file:
            source = format_source_info(source_file, "xyz")

        partitioning = format_partitioning_info(
            algorithm=self.method,
            n_fragments=self.n_fragments,
            tiers=self.tiers,
            n_primary=self.n_primary,
            n_secondary=self.n_secondary,
            n_tertiary=self.n_tertiary,
        )

        return FragmentTree(
            fragments=fragments,
            source=source,
            partitioning=partitioning,
        )

    def _build_partition_tiered(self, molecules: List[Molecule]) -> TieredPartitionResult:
        """Build the tiered partition with tuple labels."""
        centroids = compute_centroids(molecules)

        if self.method == "geom_planes":
            return self._partition_geometric_tiered(molecules, centroids)
        return self._partition_clustering_tiered(molecules, centroids)

    def _partition_geometric_tiered(
        self,
        molecules: List[Molecule],
        centroids: np.ndarray,
    ) -> TieredPartitionResult:
        """Partition using geometric planes (tiered mode)."""
        assert self.tiers is not None
        assert self.n_primary is not None
        assert self.n_secondary is not None
        n_t = self.n_tertiary if self.tiers == 3 else 1
        prim, sec, ter = partition_by_planes_tiered(
            centroids, self.n_primary, self.n_secondary, n_t or 1
        )

        if self.tiers == 2:
            labels: List[Union[LabelTuple2, LabelTuple3]] = [
                (int(p), int(s)) for p, s in zip(prim, sec)
            ]
        else:
            labels = [
                (int(p), int(s), int(t)) for p, s, t in zip(prim, sec, ter)
            ]

        return TieredPartitionResult(
            tiers=self.tiers,
            n_primary=self.n_primary,
            n_secondary=self.n_secondary,
            n_tertiary=self.n_tertiary if self.tiers == 3 else None,
            labels=labels,
            centroids=centroids,
            molecules=molecules,
        )

    def _partition_clustering_tiered(
        self,
        molecules: List[Molecule],
        centroids: np.ndarray,
    ) -> TieredPartitionResult:
        """Partition using clustering algorithms (tiered mode)."""
        assert self.tiers is not None
        assert self.n_primary is not None
        assert self.n_secondary is not None
        primary_labels = partition_labels(
            centroids, self.n_primary, self.method, self.random_state,
            init=self._init_primary,
        )

        if self.tiers == 2:
            labels: List[Union[LabelTuple2, LabelTuple3]] = [(0, 0)] * len(molecules)

            for p in range(self.n_primary):
                idx = np.where(primary_labels == p)[0]
                if len(idx) == 0:
                    raise PartitionError(f"Primary cluster {p} is empty")

                sec = partition_labels(
                    centroids[idx], self.n_secondary, self.method, self.random_state,
                    init=self._init_secondary,
                )
                for local_k, wi in enumerate(idx):
                    labels[int(wi)] = (int(p), int(sec[local_k]))

            return TieredPartitionResult(
                tiers=self.tiers,
                n_primary=self.n_primary,
                n_secondary=self.n_secondary,
                n_tertiary=None,
                labels=labels,
                centroids=centroids,
                molecules=molecules,
            )

        # 3-tier partitioning
        assert self.n_tertiary is not None
        labels3: List[Union[LabelTuple2, LabelTuple3]] = [(0, 0, 0)] * len(molecules)
        n_t = self.n_tertiary

        for p in range(self.n_primary):
            idx_p = np.where(primary_labels == p)[0]
            if len(idx_p) == 0:
                raise PartitionError(f"Primary cluster {p} is empty")

            sec_p = partition_labels(
                centroids[idx_p], self.n_secondary, self.method, self.random_state,
                init=self._init_secondary,
            )

            for s in range(self.n_secondary):
                idx_ps = idx_p[np.where(sec_p == s)[0]]
                if len(idx_ps) == 0:
                    raise PartitionError(f"Secondary cluster {p}:{s} is empty")

                ter_ps = partition_labels(
                    centroids[idx_ps], n_t, self.method, self.random_state,
                    init=self._init_tertiary,
                )
                for local_k, wi in enumerate(idx_ps):
                    labels3[int(wi)] = (int(p), int(s), int(ter_ps[local_k]))

        return TieredPartitionResult(
            tiers=self.tiers,
            n_primary=self.n_primary,
            n_secondary=self.n_secondary,
            n_tertiary=n_t,
            labels=labels3,
            centroids=centroids,
            molecules=molecules,
        )

    def _validate_tiered_hierarchy(self, result: TieredPartitionResult) -> None:
        """Validate that cluster sizes are equal at every tier level."""
        assert self.n_primary is not None
        assert self.n_secondary is not None
        n_molecules = len(result.molecules)

        if self.tiers == 2:
            expected_primary = n_molecules / self.n_primary
            expected_secondary = n_molecules / (self.n_primary * self.n_secondary)

            if expected_primary % 1 or expected_secondary % 1:
                raise PartitionError(
                    f"Non-integer cluster sizes for 2-tier hierarchy: "
                    f"{n_molecules} molecules / {self.n_primary} primary "
                    f"/ {self.n_secondary} secondary"
                )

            expected_primary = int(expected_primary)
            expected_secondary = int(expected_secondary)

            for p in range(self.n_primary):
                count_p = sum(1 for lab in result.labels if lab[0] == p)
                if count_p != expected_primary:
                    raise PartitionError(
                        f"Primary cluster {p}: size {count_p} != expected {expected_primary}"
                    )

                for s in range(self.n_secondary):
                    count_ps = sum(
                        1 for lab in result.labels if lab[0] == p and lab[1] == s
                    )
                    if count_ps != expected_secondary:
                        raise PartitionError(
                            f"Secondary cluster {p}:{s}: size {count_ps} "
                            f"!= expected {expected_secondary}"
                        )

        elif self.tiers == 3:
            assert self.n_tertiary is not None
            n_t = self.n_tertiary
            expected_primary = n_molecules / self.n_primary
            expected_secondary = n_molecules / (self.n_primary * self.n_secondary)
            expected_tertiary = n_molecules / (self.n_primary * self.n_secondary * n_t)

            if expected_primary % 1 or expected_secondary % 1 or expected_tertiary % 1:
                raise PartitionError(
                    "Non-integer cluster sizes for 3-tier hierarchy"
                )

            expected_primary = int(expected_primary)
            expected_secondary = int(expected_secondary)
            expected_tertiary = int(expected_tertiary)

            for p in range(self.n_primary):
                count_p = sum(1 for lab in result.labels if lab[0] == p)
                if count_p != expected_primary:
                    raise PartitionError(
                        f"Primary cluster {p}: size {count_p} != expected {expected_primary}"
                    )

                for s in range(self.n_secondary):
                    count_ps = sum(
                        1 for lab in result.labels if lab[0] == p and lab[1] == s
                    )
                    if count_ps != expected_secondary:
                        raise PartitionError(
                            f"Secondary cluster {p}:{s}: size {count_ps} "
                            f"!= expected {expected_secondary}"
                        )

                    for t in range(n_t):
                        count_pst = sum(
                            1 for lab in result.labels
                            if len(lab) > 2 and lab[0] == p and lab[1] == s and lab[2] == t
                        )
                        if count_pst != expected_tertiary:
                            raise PartitionError(
                                f"Tertiary cluster {p}:{s}:{t}: size {count_pst} "
                                f"!= expected {expected_tertiary}"
                            )

    def _build_tiered_fragments(self, result: TieredPartitionResult) -> List[Fragment]:
        """Build hierarchical Fragment objects from tiered partition result."""
        assert self.n_primary is not None
        assert self.n_secondary is not None
        fragments: List[Fragment] = []

        if self.tiers == 2:
            for p in range(self.n_primary):
                pf_id = f"PF{p + 1}"
                pf = Fragment(id=pf_id)

                for s in range(self.n_secondary):
                    sf_id = f"{pf_id}_SF{s + 1}"
                    chosen = [
                        result.molecules[i]
                        for i, lab in enumerate(result.labels)
                        if lab[0] == p and lab[1] == s
                    ]
                    sf = Fragment.from_molecules(chosen, sf_id)
                    sf.metadata["n_molecules"] = len(chosen)
                    pf.fragments.append(sf)

                fragments.append(pf)

        elif self.tiers == 3:
            assert self.n_tertiary is not None
            n_t = self.n_tertiary
            for p in range(self.n_primary):
                pf_id = f"PF{p + 1}"
                pf = Fragment(id=pf_id)

                for s in range(self.n_secondary):
                    sf_id = f"{pf_id}_SF{s + 1}"
                    sf = Fragment(id=sf_id)

                    for t in range(n_t):
                        tf_id = f"{sf_id}_TF{t + 1}"
                        chosen = [
                            result.molecules[i]
                            for i, lab in enumerate(result.labels)
                            if len(lab) > 2 and lab[0] == p and lab[1] == s and lab[2] == t
                        ]
                        tf = Fragment.from_molecules(chosen, tf_id)
                        tf.metadata["n_molecules"] = len(chosen)
                        sf.fragments.append(tf)

                    pf.fragments.append(sf)

                fragments.append(pf)

        return fragments
