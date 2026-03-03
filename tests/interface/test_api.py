# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment public API."""

import tempfile
from pathlib import Path


class TestPublicAPI:
    """Tests for the public API."""

    def test_import_autofragment(self):
        """Test that autofragment can be imported."""
        import autofragment as af
        assert hasattr(af, "__version__")
        assert hasattr(af, "partition_xyz")
        assert hasattr(af, "MolecularPartitioner")
        assert hasattr(af, "BatchPartitioner")

    def test_import_types(self):
        """Test that core types can be imported."""
        from autofragment import Atom, Fragment, FragmentTree
        assert Atom is not None
        assert Fragment is not None
        assert FragmentTree is not None

    def test_import_io(self):
        """Test that io module can be imported."""
        from autofragment import io
        assert hasattr(io, "read_xyz")
        assert hasattr(io, "read_xyz_molecules")

    def test_partition_xyz_convenience(self, water16_path):
        """Test the partition_xyz convenience function."""
        import autofragment as af

        tree = af.partition_xyz(str(water16_path), n_fragments=4, method="kmeans")

        assert tree.n_fragments == 4
        assert len(tree.fragments) == 4

    def test_partition_xyz_output(self, water16_path):
        """Test that partition_xyz produces valid JSON output."""
        import autofragment as af

        tree = af.partition_xyz(str(water16_path), n_fragments=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"
            tree.to_json(path)

            # Read back
            loaded = af.FragmentTree.from_json(path)
            assert loaded.n_fragments == 4

    def test_full_workflow(self, water16_path):
        """Test a complete workflow."""
        import autofragment as af

        # Read molecules
        system = af.io.read_xyz(str(water16_path))
        molecules = system.to_molecules(require_metadata=True)
        assert len(molecules) == 16

        # Create partitioner
        partitioner = af.MolecularPartitioner(
            n_fragments=4,
        )

        # Partition
        tree = partitioner.partition(system, source_file=str(water16_path))

        # Check structure
        assert tree.n_fragments == 4
        for frag in tree.fragments:
            assert frag.n_atoms > 0

        # Export
        data = tree.to_dict()
        assert "fragments" in data
        assert "partitioning" in data
