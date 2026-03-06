# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Batch partitioner for consistent labeling across multiple files.

This module provides the BatchPartitioner class for partitioning multiple
molecular systems with consistent fragment labeling, using a reference
structure for alignment and matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from autofragment.algorithms.hungarian import hungarian_assignment
from autofragment.core.geometry import (
    compute_centroids,
    hybrid_distance,
    kabsch_align,
    molecule_rmsd,
)
from autofragment.core.types import (
    ChemicalSystem,
    Fragment,
    FragmentTree,
    Molecule,
    molecule_to_coords,
    system_to_molecules,
)
from autofragment.io.output import format_partitioning_info, format_source_info
from autofragment.io.xyz import read_xyz
from autofragment.partitioners.molecular import MolecularPartitioner


def _labels_from_tree(tree: "FragmentTree", molecules: List[Molecule]) -> List[int]:
    """Reconstruct per-molecule labels from a FragmentTree.

    Each fragment stores a contiguous block of atoms built from one or more
    molecules.  We match fragment atoms back to original molecules by
    coordinate comparison to recover the label assignment.
    """
    n_mol = len(molecules)
    labels = [0] * n_mol

    # Pre-compute centroid for each molecule for fast matching
    mol_centroids = np.array([
        np.mean([a.coords for a in mol], axis=0) for mol in molecules
    ])

    for frag_idx, frag in enumerate(tree.fragments):
        n_mol_in_frag = frag.metadata.get("n_molecules", 0)
        if n_mol_in_frag == 0:
            continue

        frag_coords = frag.get_coords()
        # Figure out atoms_per_molecule from total atoms / n_molecules
        atoms_per_mol = frag.n_atoms // n_mol_in_frag if n_mol_in_frag else 0
        if atoms_per_mol == 0:
            continue

        # For each molecule-sized chunk in the fragment, match to original
        for chunk_start in range(0, frag.n_atoms, atoms_per_mol):
            chunk = frag_coords[chunk_start:chunk_start + atoms_per_mol]
            chunk_centroid = chunk.mean(axis=0)
            # Find the closest original molecule
            dists = np.linalg.norm(mol_centroids - chunk_centroid, axis=1)
            best_mol = int(np.argmin(dists))
            labels[best_mol] = frag_idx

    return labels


class BatchPartitionError(ValueError):
    """Raised when batch partitioning fails."""

    pass


@dataclass(frozen=True)
class AssignmentQuality:
    """Quality metrics for molecule-to-reference assignment."""

    mean_centroid_distance: float
    max_centroid_distance: float
    mean_rmsd: float
    max_rmsd: float
    std_centroid_distance: float
    std_rmsd: float

    def is_good(
        self,
        centroid_threshold: float = 2.0,
        rmsd_threshold: float = 2.5,
    ) -> bool:
        """Check if the assignment quality is acceptable."""
        return (
            self.mean_centroid_distance < centroid_threshold
            and self.max_centroid_distance < centroid_threshold * 2.5
            and self.mean_rmsd < rmsd_threshold
            and self.max_rmsd < rmsd_threshold * 2.5
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"  Centroid: mean={self.mean_centroid_distance:.4f} A, "
            f"max={self.max_centroid_distance:.4f} A\n"
            f"  RMSD: mean={self.mean_rmsd:.4f} A, max={self.max_rmsd:.4f} A"
        )


class BatchPartitioner:
    """
    Partitioner for multiple files with consistent labeling.

    This partitioner uses a reference structure to establish fragment labels,
    then applies those labels to additional structures by matching molecules
    based on their positions.

    Parameters
    ----------
    reference_tree : FragmentTree
        Reference fragment tree (from MolecularPartitioner).
    reference_molecules : ChemicalSystem | Sequence[Molecule]
        Reference system used to create the tree.
    reference_labels : Sequence
        Labels from reference partitioning.
    metric : str, optional
        Distance metric for matching: "centroid", "rmsd", or "hybrid".
        Default is "rmsd".
    hybrid_alpha : float, optional
        Weight for centroid in hybrid metric. Default is 0.5.
    align : bool, optional
        If True, apply Kabsch alignment before matching. Default is False.
    centroid_threshold : float, optional
        Quality threshold for centroid distance. Default is 2.0.
    rmsd_threshold : float, optional
        Quality threshold for RMSD. Default is 2.5.

    Examples
    --------
    >>> partitioner = MolecularPartitioner(n_fragments=4)
    >>> reference = read_xyz("reference.xyz")
    >>> ref_tree = partitioner.partition(reference)
    >>>
    >>> batch = BatchPartitioner.from_partitioner(
    ...     partitioner, reference, metric="rmsd", align=True
    ... )
    >>> for filepath in trajectory_files:
    ...     result = batch.partition_file(filepath)
    ...     result.to_json(f"output/{filepath.stem}.json")
    """

    def __init__(
        self,
        reference_tree: FragmentTree,
        reference_molecules: Sequence[Molecule],
        reference_labels: Sequence,
        *,
        n_fragments: int,
        metric: str = "rmsd",
        hybrid_alpha: float = 0.5,
        align: bool = False,
        centroid_threshold: float = 2.0,
        rmsd_threshold: float = 2.5,
        method: str = "kmeans_constrained",
        tiers: Optional[int] = None,
        n_primary: Optional[int] = None,
        n_secondary: Optional[int] = None,
        n_tertiary: Optional[int] = None,
    ):
        """Initialize a new BatchPartitioner instance."""
        self.reference_tree = reference_tree
        self.reference_molecules = list(reference_molecules)
        self.reference_labels = list(reference_labels)
        self.n_fragments = n_fragments
        self.metric = metric
        self.hybrid_alpha = hybrid_alpha
        self.align = align
        self.centroid_threshold = centroid_threshold
        self.rmsd_threshold = rmsd_threshold
        self.method = method
        self.tiers = tiers
        self.n_primary = n_primary
        self.n_secondary = n_secondary
        self.n_tertiary = n_tertiary

    @classmethod
    def from_partitioner(
        cls,
        partitioner: MolecularPartitioner,
        reference_molecules: ChemicalSystem | Sequence[Molecule],
        *,
        metric: str = "rmsd",
        hybrid_alpha: float = 0.5,
        align: bool = False,
        centroid_threshold: float = 2.0,
        rmsd_threshold: float = 2.5,
        source_file: Optional[str] = None,
    ) -> "BatchPartitioner":
        """
        Create a BatchPartitioner from a MolecularPartitioner.

        This is a convenience method that creates the reference partition
        and extracts the necessary labels.

        Parameters
        ----------
        partitioner : MolecularPartitioner
            The partitioner to use for reference.
        reference_molecules : ChemicalSystem | Sequence[Molecule]
            Reference system to partition.
        metric : str, optional
            Distance metric for matching. Default is "rmsd".
        hybrid_alpha : float, optional
            Weight for centroid in hybrid metric. Default is 0.5.
        align : bool, optional
            If True, apply Kabsch alignment. Default is False.
        centroid_threshold : float, optional
            Quality threshold for centroid distance. Default is 2.0.
        rmsd_threshold : float, optional
            Quality threshold for RMSD. Default is 2.5.
        source_file : str, optional
            Path to reference file for metadata.

        Returns
        -------
        BatchPartitioner
            Configured batch partitioner.
        """
        # Build reference partition
        if isinstance(reference_molecules, ChemicalSystem):
            reference_system = reference_molecules
        else:
            reference_system = ChemicalSystem.from_molecules(reference_molecules)

        ref_molecules = system_to_molecules(reference_system, require_metadata=True)
        ref_tree = partitioner.partition(reference_system, source_file)

        if partitioner.tiers is not None:
            # Tiered mode: get tuple labels from tiered partition result
            tiered_result = partitioner._build_partition_tiered(list(ref_molecules))
            ref_labels: list = tiered_result.labels
        else:
            # Flat mode: derive labels from finalized tree (post-refinement)
            ref_labels = _labels_from_tree(ref_tree, ref_molecules)

        return cls(
            reference_tree=ref_tree,
            reference_molecules=ref_molecules,
            reference_labels=ref_labels,
            n_fragments=partitioner.n_fragments,
            metric=metric,
            hybrid_alpha=hybrid_alpha,
            align=align,
            centroid_threshold=centroid_threshold,
            rmsd_threshold=rmsd_threshold,
            method=partitioner.method,
            tiers=partitioner.tiers,
            n_primary=partitioner.n_primary,
            n_secondary=partitioner.n_secondary,
            n_tertiary=partitioner.n_tertiary,
        )

    def partition(
        self,
        system: ChemicalSystem,
        source_file: Optional[str] = None,
        force: bool = False,
    ) -> FragmentTree:
        """
        Partition a system using reference labels.

        Parameters
        ----------
        system : ChemicalSystem
            Chemical system to partition.
        source_file : str, optional
            Path to source file for metadata.
        force : bool, optional
            If True, proceed even with poor assignment quality. Default is False.

        Returns
        -------
        FragmentTree
            The partitioned fragment tree.

        Raises
        ------
        BatchPartitionError
            If molecule count doesn't match or assignment quality is poor.
        """
        molecules = system_to_molecules(system, require_metadata=True)

        if len(molecules) != len(self.reference_molecules):
            raise BatchPartitionError(
                f"Molecule count mismatch: {len(molecules)} vs "
                f"{len(self.reference_molecules)} (reference)"
            )

        # Apply alignment if requested
        if self.align:
            molecules = kabsch_align(self.reference_molecules, molecules)

        # Match molecules to reference
        assignment = self._match_molecules(molecules)

        # Validate assignment quality
        quality = self._validate_assignment(molecules, assignment)
        if not quality.is_good(self.centroid_threshold, self.rmsd_threshold):
            if not force:
                raise BatchPartitionError(
                    f"Poor assignment quality:\n{quality.summary()}\n"
                    f"Use force=True to proceed anyway."
                )

        # Apply reference labels
        target_labels = [
            self.reference_labels[assignment[i]] for i in range(len(molecules))
        ]

        # Build fragments
        if self.tiers is not None:
            fragments = self._build_tiered_fragments(molecules, target_labels)
        else:
            fragments = self._build_fragments(molecules, target_labels)

        # Build metadata
        source = {}
        if source_file:
            source = format_source_info(source_file, "xyz")

        partitioning_kwargs: Dict[str, Any] = {}
        if self.tiers is not None:
            partitioning_kwargs.update(
                tiers=self.tiers,
                n_primary=self.n_primary,
                n_secondary=self.n_secondary,
                n_tertiary=self.n_tertiary,
            )

        partitioning = format_partitioning_info(
            algorithm=self.method,
            n_fragments=self.n_fragments,
            **partitioning_kwargs,
        )

        return FragmentTree(
            fragments=fragments,
            source=source,
            partitioning=partitioning,
        )

    def partition_file(
        self,
        filepath: str | Path,
        force: bool = False,
        xyz_units: Literal["angstrom", "bohr"] = "angstrom",
    ) -> FragmentTree:
        """
        Partition molecules from a file.

        Parameters
        ----------
        filepath : str or Path
            Path to XYZ file.
        force : bool, optional
            If True, proceed even with poor assignment quality. Default is False.
        xyz_units : str, optional
            Units for XYZ coordinates. Default is "angstrom".

        Returns
        -------
        FragmentTree
            The partitioned fragment tree.
        """
        system = read_xyz(filepath, xyz_units=xyz_units)
        return self.partition(system, source_file=str(filepath), force=force)

    def partition_many(
        self,
        systems: Sequence[ChemicalSystem],
        *,
        source_files: Optional[Sequence[str]] = None,
        force: bool = False,
        n_jobs: Optional[int] = None,
        backend: str = "loky",
    ) -> List[FragmentTree]:
        """
        Partition multiple systems in parallel.

        Parameters
        ----------
        systems : Sequence[ChemicalSystem]
            Chemical systems to partition.
        source_files : Sequence[str], optional
            Corresponding source file paths for metadata.
        force : bool, optional
            If True, proceed even with poor assignment quality. Default is False.
        n_jobs : int, optional
            Number of parallel jobs. None uses all CPUs via joblib, or falls
            back to sequential if joblib is not installed.
        backend : str, optional
            Joblib backend. Default is "loky".

        Returns
        -------
        List[FragmentTree]
            Fragment trees for each system.
        """
        from autofragment.utils.parallel import parallel_map

        srcs = source_files or [None] * len(systems)

        def _do(args: tuple) -> FragmentTree:
            sys, src = args
            return self.partition(sys, source_file=src, force=force)

        return parallel_map(
            _do, list(zip(systems, srcs)), n_jobs=n_jobs, backend=backend
        )

    def partition_files(
        self,
        filepaths: Sequence[str | Path],
        *,
        force: bool = False,
        xyz_units: Literal["angstrom", "bohr"] = "angstrom",
        n_jobs: Optional[int] = None,
        backend: str = "loky",
    ) -> List[FragmentTree]:
        """
        Partition multiple XYZ files in parallel.

        Parameters
        ----------
        filepaths : Sequence[str | Path]
            Paths to XYZ files.
        force : bool, optional
            If True, proceed even with poor assignment quality. Default is False.
        xyz_units : str, optional
            Units for XYZ coordinates. Default is "angstrom".
        n_jobs : int, optional
            Number of parallel jobs. None uses all CPUs via joblib, or falls
            back to sequential if joblib is not installed.
        backend : str, optional
            Joblib backend. Default is "loky".

        Returns
        -------
        List[FragmentTree]
            Fragment trees for each file.
        """
        from autofragment.utils.parallel import parallel_map

        def _do(fp: str | Path) -> FragmentTree:
            return self.partition_file(fp, force=force, xyz_units=xyz_units)

        return parallel_map(
            _do, list(filepaths), n_jobs=n_jobs, backend=backend
        )

    def _match_molecules(self, target: List[Molecule]) -> Dict[int, int]:
        """Match target molecules to reference using optimal assignment."""
        n = len(self.reference_molecules)
        cost = np.zeros((n, n), dtype=float)

        if self.metric == "centroid":
            ref_cents = compute_centroids(self.reference_molecules)
            tgt_cents = compute_centroids(target)
            diff = ref_cents[:, np.newaxis, :] - tgt_cents[np.newaxis, :, :]
            cost = np.linalg.norm(diff, axis=2)

        elif self.metric == "rmsd":
            for i in range(n):
                for j in range(n):
                    cost[i, j] = molecule_rmsd(
                        self.reference_molecules[i], target[j]
                    )

        elif self.metric == "hybrid":
            for i in range(n):
                for j in range(n):
                    cost[i, j] = hybrid_distance(
                        self.reference_molecules[i],
                        target[j],
                        alpha=self.hybrid_alpha,
                    )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        ref_rows, target_cols = hungarian_assignment(cost)

        target_to_ref: Dict[int, int] = {}
        for ref_i, tgt_j in zip(ref_rows.tolist(), target_cols.tolist()):
            target_to_ref[int(tgt_j)] = int(ref_i)

        return target_to_ref

    def _validate_assignment(
        self,
        target: List[Molecule],
        assignment: Dict[int, int],
    ) -> AssignmentQuality:
        """Compute quality metrics for the assignment."""
        centroid_distances = []
        rmsds = []

        for tgt_idx, ref_idx in assignment.items():
            ref_coords = molecule_to_coords(self.reference_molecules[ref_idx])
            tgt_coords = molecule_to_coords(target[tgt_idx])

            ref_cent = ref_coords.mean(axis=0)
            tgt_cent = tgt_coords.mean(axis=0)
            centroid_distances.append(float(np.linalg.norm(ref_cent - tgt_cent)))

            rmsds.append(
                molecule_rmsd(self.reference_molecules[ref_idx], target[tgt_idx])
            )

        return AssignmentQuality(
            mean_centroid_distance=float(np.mean(centroid_distances)),
            max_centroid_distance=float(np.max(centroid_distances)),
            mean_rmsd=float(np.mean(rmsds)),
            max_rmsd=float(np.max(rmsds)),
            std_centroid_distance=float(np.std(centroid_distances)),
            std_rmsd=float(np.std(rmsds)),
        )

    def _build_fragments(
        self,
        molecules: List[Molecule],
        labels: List,
    ) -> List[Fragment]:
        """Build Fragment objects from molecules and labels (flat mode)."""
        fragments: List[Fragment] = []

        for k in range(self.n_fragments):
            chosen = [
                molecules[i]
                for i, lbl in enumerate(labels)
                if int(lbl) == k
            ]
            fragments.append(Fragment.from_molecules(chosen, f"F{k + 1}"))

        return fragments

    def _build_tiered_fragments(
        self,
        molecules: List[Molecule],
        labels: List,
    ) -> List[Fragment]:
        """Build hierarchical Fragment objects from tiered labels."""
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
                        molecules[i]
                        for i, lab in enumerate(labels)
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
                            molecules[i]
                            for i, lab in enumerate(labels)
                            if lab[0] == p and lab[1] == s and lab[2] == t
                        ]
                        tf = Fragment.from_molecules(chosen, tf_id)
                        tf.metadata["n_molecules"] = len(chosen)
                        sf.fragments.append(tf)

                    pf.fragments.append(sf)

                fragments.append(pf)

        return fragments
