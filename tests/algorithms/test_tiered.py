# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for tiered (hierarchical) partitioning functionality."""

import pytest

from autofragment import io
from autofragment.core.types import FragmentTree
from autofragment.partitioners.batch import BatchPartitioner
from autofragment.partitioners.molecular import (
    MolecularPartitioner,
    PartitionError,
)


@pytest.fixture
def water16_system():
    """Load water16 test fixture as a ChemicalSystem."""
    return io.read_xyz("tests/fixtures/water16.xyz")


class TestTiered2Tier:
    """Tests for 2-tier hierarchical partitioning."""

    def test_basic_2tier(self, water16_system):
        """2-tier partitioning produces correct hierarchy."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 2
        assert tree._is_hierarchical
        for pf in tree.fragments:
            assert not pf.is_leaf
            assert len(pf.fragments) == 2
            for sf in pf.fragments:
                assert sf.is_leaf
                assert sf.n_atoms > 0

    def test_2tier_atom_count(self, water16_system):
        """All atoms preserved across 2-tier hierarchy."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        tree = part.partition(water16_system)
        assert tree.n_atoms == 48  # 16 water molecules * 3 atoms each

    def test_2tier_n_fragments(self, water16_system):
        """n_fragments counts all fragments recursively."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        tree = part.partition(water16_system)
        # 2 primary + 4 secondary = 6 total
        assert tree.n_fragments == 6

    def test_2tier_fragment_naming(self, water16_system):
        """Fragments follow PF/SF naming convention."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        tree = part.partition(water16_system)

        assert tree.fragments[0].id == "PF1"
        assert tree.fragments[1].id == "PF2"
        assert tree.fragments[0].fragments[0].id == "PF1_SF1"
        assert tree.fragments[0].fragments[1].id == "PF1_SF2"

    def test_2tier_4x4(self, water16_system):
        """4x4 2-tier partitioning works."""
        part = MolecularPartitioner(
            tiers=2, n_primary=4, n_secondary=4, method="kmeans"
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 4
        for pf in tree.fragments:
            assert len(pf.fragments) == 4
        assert tree.n_atoms == 48


class TestTiered3Tier:
    """Tests for 3-tier hierarchical partitioning."""

    def test_basic_3tier(self, water16_system):
        """3-tier partitioning produces correct hierarchy."""
        part = MolecularPartitioner(
            tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 2
        assert tree._is_hierarchical
        for pf in tree.fragments:
            assert not pf.is_leaf
            assert len(pf.fragments) == 2
            for sf in pf.fragments:
                assert not sf.is_leaf
                assert len(sf.fragments) == 2
                for tf in sf.fragments:
                    assert tf.is_leaf
                    assert tf.n_atoms > 0

    def test_3tier_atom_count(self, water16_system):
        """All atoms preserved across 3-tier hierarchy."""
        part = MolecularPartitioner(
            tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
        )
        tree = part.partition(water16_system)
        assert tree.n_atoms == 48

    def test_3tier_n_fragments(self, water16_system):
        """n_fragments counts all fragments recursively."""
        part = MolecularPartitioner(
            tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
        )
        tree = part.partition(water16_system)
        # 2 primary + 4 secondary + 8 tertiary = 14 total
        assert tree.n_fragments == 14

    def test_3tier_fragment_naming(self, water16_system):
        """Fragments follow PF/SF/TF naming convention."""
        part = MolecularPartitioner(
            tiers=3, n_primary=2, n_secondary=2, n_tertiary=2, method="kmeans"
        )
        tree = part.partition(water16_system)

        assert tree.fragments[0].id == "PF1"
        sf = tree.fragments[0].fragments[0]
        assert sf.id == "PF1_SF1"
        tf = sf.fragments[0]
        assert tf.id == "PF1_SF1_TF1"


class TestTieredValidation:
    """Tests for tiered validation and error handling."""

    def test_invalid_tiers(self):
        """tiers must be 2 or 3."""
        with pytest.raises(ValueError, match="tiers must be 2 or 3"):
            MolecularPartitioner(tiers=1, n_primary=2, n_secondary=2)

    def test_missing_n_primary(self):
        """n_primary required for tiered mode."""
        with pytest.raises(ValueError, match="n_primary and n_secondary are required"):
            MolecularPartitioner(tiers=2, n_secondary=2)

    def test_missing_n_tertiary_3tier(self):
        """n_tertiary required for 3-tier mode."""
        with pytest.raises(ValueError, match="n_tertiary is required"):
            MolecularPartitioner(tiers=3, n_primary=2, n_secondary=2)

    def test_strict_balanced_validates_sizes(self, water16_system):
        """strict_balanced validates equal cluster sizes."""
        # 16 molecules / (3 primary * 2 secondary) = non-integer
        part = MolecularPartitioner(
            tiers=2, n_primary=3, n_secondary=2,
            method="kmeans_constrained", strict_balanced=True,
        )
        with pytest.raises(PartitionError, match="Non-integer"):
            part.partition(water16_system)


class TestFlatModeRegression:
    """Ensure flat mode still works identically."""

    def test_flat_mode_default(self, water16_system):
        """Default (tiers=None) produces flat fragments."""
        part = MolecularPartitioner(n_fragments=4, method="kmeans")
        tree = part.partition(water16_system)

        assert not tree._is_hierarchical
        assert tree.n_fragments == 4
        assert all(f.is_leaf for f in tree.fragments)
        assert all(f.fragments == [] for f in tree.fragments)

    def test_flat_mode_atom_count(self, water16_system):
        """Flat mode preserves all atoms."""
        part = MolecularPartitioner(n_fragments=4, method="kmeans")
        tree = part.partition(water16_system)
        assert tree.n_atoms == 48


class TestTieredWithSeeding:
    """Tests for tiered mode with custom seeding strategies."""

    def test_tiered_with_pca_seeding(self, water16_system):
        """Tiered partitioning with init_strategy='pca' produces valid result."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2,
            method="kmeans", init_strategy="pca",
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 2
        assert tree._is_hierarchical
        assert tree.n_atoms == 48

    def test_tiered_with_per_tier_seeding(self, water16_system):
        """Per-tier seeding overrides work."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2,
            method="kmeans",
            init_strategy="pca",
            init_strategy_secondary="radial",
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 2
        assert tree.n_atoms == 48

    def test_flat_with_seeding(self, water16_system):
        """Flat mode with init_strategy works."""
        part = MolecularPartitioner(
            n_fragments=4, method="kmeans", init_strategy="pca",
        )
        tree = part.partition(water16_system)

        assert not tree._is_hierarchical
        assert tree.n_fragments == 4


class TestTieredGeometric:
    """Tests for tiered mode with geometric plane method."""

    def test_tiered_geom_planes_2tier(self, water16_system):
        """Geometric planes work in 2-tier mode."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="geom_planes"
        )
        tree = part.partition(water16_system)

        assert tree.n_primary == 2
        assert tree._is_hierarchical
        assert tree.n_atoms == 48


class TestTieredBatch:
    """Tests for BatchPartitioner with tiered MolecularPartitioner."""

    def test_batch_from_tiered_partitioner(self, water16_system):
        """BatchPartitioner.from_partitioner works with tiered partitioner."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        batch = BatchPartitioner.from_partitioner(part, water16_system)

        assert batch.tiers == 2
        assert batch.n_primary == 2
        assert batch.n_secondary == 2

    def test_batch_tiered_partition(self, water16_system):
        """Tiered batch partition produces matching hierarchy."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        batch = BatchPartitioner.from_partitioner(part, water16_system)

        # Partition the same system (should produce consistent labeling)
        tree = batch.partition(water16_system, force=True)

        assert tree._is_hierarchical
        assert tree.n_primary == 2
        for pf in tree.fragments:
            assert len(pf.fragments) == 2
        assert tree.n_atoms == 48

    def test_batch_flat_regression(self, water16_system):
        """Flat batch still works."""
        part = MolecularPartitioner(n_fragments=4, method="kmeans")
        batch = BatchPartitioner.from_partitioner(part, water16_system)

        assert batch.tiers is None
        tree = batch.partition(water16_system, force=True)
        assert not tree._is_hierarchical
        assert len(tree.fragments) == 4


class TestTieredJsonRoundtrip:
    """Tests for JSON serialization/deserialization of tiered trees."""

    def test_tiered_json_roundtrip(self, water16_system, tmp_path):
        """Tiered tree survives JSON roundtrip."""
        part = MolecularPartitioner(
            tiers=2, n_primary=2, n_secondary=2, method="kmeans"
        )
        tree = part.partition(water16_system)

        json_path = tmp_path / "tiered.json"
        tree.to_json(json_path)
        loaded = FragmentTree.from_json(json_path)

        assert loaded.n_primary == tree.n_primary
        assert loaded.n_fragments == tree.n_fragments
        assert loaded.n_atoms == tree.n_atoms
        assert loaded._is_hierarchical

        # Verify structure matches
        for orig_pf, load_pf in zip(tree.fragments, loaded.fragments):
            assert orig_pf.id == load_pf.id
            assert len(orig_pf.fragments) == len(load_pf.fragments)
            for orig_sf, load_sf in zip(orig_pf.fragments, load_pf.fragments):
                assert orig_sf.id == load_sf.id
                assert orig_sf.n_atoms == load_sf.n_atoms
