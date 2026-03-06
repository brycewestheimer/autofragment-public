# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment.partitioners module."""

import tempfile
from pathlib import Path

import pytest

from autofragment import io
from autofragment.core.types import Atom, ChemicalSystem, molecules_to_system
from autofragment.partitioners.batch import BatchPartitioner
from autofragment.partitioners.molecular import MolecularPartitioner


class TestMolecularPartitioner:
    """Tests for MolecularPartitioner."""

    def test_partition_flat(self, water16_path):
        """Test flat partitioning."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4,
            method="kmeans",
        )
        tree = partitioner.partition(system, source_file=str(water16_path))

        assert tree.n_fragments == 4
        for frag in tree.fragments:
            assert frag.n_atoms > 0

    def test_partition_constrained(self, water16_path):
        """Test flat partitioning with constrained k-means."""
        system = io.read_xyz(water16_path)
        try:
            partitioner = MolecularPartitioner(
                n_fragments=4,
                method="kmeans_constrained",
            )
            tree = partitioner.partition(system)
            assert tree.n_fragments == 4
        except ImportError:
            pytest.skip("k-means-constrained not installed")

    def test_partition_with_source_metadata(self, water16_path):
        """Test that source metadata is included."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(n_fragments=4)
        tree = partitioner.partition(system, source_file=str(water16_path))

        assert tree.source["file"] == "water16.xyz"
        assert tree.source["format"] == "xyz"
        assert tree.partitioning["n_fragments"] == 4

    def test_output_json_format(self, water16_path):
        """Test that output JSON has correct format."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(n_fragments=4)
        tree = partitioner.partition(system, source_file=str(water16_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"
            tree.to_json(path)

            import json
            data = json.loads(path.read_text())

            assert "version" in data
            assert "source" in data
            assert "partitioning" in data
            assert "fragments" in data
            assert "interfragment_bonds" in data

    def test_reproducibility(self, water16_path):
        """Test that partitioning is reproducible."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, random_state=42
        )

        tree1 = partitioner.partition(system)
        tree2 = partitioner.partition(system)

        # Check same structure
        assert len(tree1.fragments) == len(tree2.fragments)
        for f1, f2 in zip(tree1.fragments, tree2.fragments):
            assert f1.id == f2.id
            assert f1.n_atoms == f2.n_atoms

    def test_topology_refine_runs(self):
        """Test optional topology refinement path after clustering."""
        atoms = [
            Atom(symbol="C", coords=[0.0, 0.0, 0.0]),
            Atom(symbol="C", coords=[1.5, 0.0, 0.0]),
            Atom(symbol="C", coords=[3.0, 0.0, 0.0]),
            Atom(symbol="C", coords=[4.5, 0.0, 0.0]),
        ]
        system = ChemicalSystem(
            atoms=atoms,
            bonds=[
                {"atom1": 0, "atom2": 1, "order": 1.0},
                {"atom1": 1, "atom2": 2, "order": 1.0},
                {"atom1": 2, "atom2": 3, "order": 1.0},
            ],
            metadata={
                "atoms_per_molecule": 1,
                "molecule_atom_indices": [[0], [1], [2], [3]],
            },
        )

        partitioner = MolecularPartitioner(
            n_fragments=2,
            method="kmeans",
            random_state=42,
            topology_refine=True,
            topology_mode="graph",
            topology_hops=1,
        )
        tree = partitioner.partition(system)

        assert tree.n_fragments == 2
        assert sum(fragment.n_atoms for fragment in tree.fragments) == 4


class TestBatchPartitioner:
    """Tests for BatchPartitioner."""

    def test_batch_same_structure(self, water16_path):
        """Test batch partitioning with same input file."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )

        batch = BatchPartitioner.from_partitioner(
            partitioner, system, metric="rmsd"
        )

        # Partition same molecules
        tree = batch.partition(system)
        assert tree.n_fragments == 4

    def test_batch_mismatched_count(self, water16_path):
        """Test that mismatched molecule count raises error."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(n_fragments=4)

        batch = BatchPartitioner.from_partitioner(partitioner, system)

        # Try to partition with different count
        from autofragment.partitioners.batch import BatchPartitionError
        with pytest.raises(BatchPartitionError, match="count mismatch"):
            molecules = system.to_molecules(require_metadata=True)
            smaller_system = molecules_to_system(molecules[:8])
            batch.partition(smaller_system)

    def test_batch_with_alignment(self, water16_path):
        """Test batch partitioning with Kabsch alignment."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )

        batch = BatchPartitioner.from_partitioner(
            partitioner, system, align=True
        )

        tree = batch.partition(system)
        assert tree.n_fragments == 4

    @pytest.mark.parametrize("metric", ["centroid", "rmsd", "hybrid"])
    def test_batch_metrics(self, water16_path, metric):
        """Test different distance metrics for batch matching."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )

        batch = BatchPartitioner.from_partitioner(
            partitioner, system, metric=metric
        )

        tree = batch.partition(system)
        assert tree.n_fragments == 4

    def test_partition_many(self, water16_path):
        """Test partition_many with multiple systems."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )
        batch = BatchPartitioner.from_partitioner(
            partitioner, system, metric="centroid"
        )

        trees = batch.partition_many(
            [system, system],
            source_files=[str(water16_path), str(water16_path)],
            n_jobs=1,
        )
        assert len(trees) == 2
        for tree in trees:
            assert tree.n_fragments == 4

    def test_partition_many_no_source_files(self, water16_path):
        """Test partition_many without source_files."""
        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )
        batch = BatchPartitioner.from_partitioner(
            partitioner, system, metric="centroid"
        )

        trees = batch.partition_many([system], n_jobs=1)
        assert len(trees) == 1
        assert trees[0].n_fragments == 4

    def test_vectorized_centroid_cost_matrix(self, water16_path):
        """Test that vectorized centroid cost matches naive loop."""
        import numpy as np
        from autofragment.core.geometry import compute_centroids

        system = io.read_xyz(water16_path)
        partitioner = MolecularPartitioner(
            n_fragments=4, method="kmeans"
        )
        batch = BatchPartitioner.from_partitioner(
            partitioner, system, metric="centroid"
        )
        molecules = system.to_molecules(require_metadata=True)

        ref_cents = compute_centroids(batch.reference_molecules)
        tgt_cents = compute_centroids(molecules)
        n = len(ref_cents)

        # Vectorized (as implemented)
        diff = ref_cents[:, np.newaxis, :] - tgt_cents[np.newaxis, :, :]
        cost_vec = np.linalg.norm(diff, axis=2)

        # Naive loop for reference
        cost_loop = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost_loop[i, j] = float(np.linalg.norm(ref_cents[i] - tgt_cents[j]))

        np.testing.assert_allclose(cost_vec, cost_loop, atol=1e-12)
