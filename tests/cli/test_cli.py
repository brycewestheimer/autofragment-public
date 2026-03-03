# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the autofragment CLI."""

import shutil
from pathlib import Path

import pytest

from autofragment.cli.main import main

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data"


class TestSingleCommand:
    """Tests for the 'single' subcommand."""

    def test_single_water16(self, tmp_path):
        """Running single on water16.xyz should succeed."""
        water = FIXTURES_DIR / "water16.xyz"
        if not water.exists():
            pytest.skip("water16.xyz fixture not available")
        output = tmp_path / "water16.json"
        rc = main(["single", "--input", str(water), "--output", str(output)])
        assert rc == 0
        assert output.exists()

    def test_single_water3(self, tmp_path):
        """Running single on water3.xyz should succeed."""
        water = DATA_DIR / "water3.xyz"
        if not water.exists():
            pytest.skip("water3.xyz data not available")
        output = tmp_path / "water3.json"
        rc = main([
            "single",
            "--input", str(water),
            "--output", str(output),
            "--n-fragments", "2",
        ])
        assert rc == 0
        assert output.exists()

    def test_single_atoms_per_molecule(self, tmp_path):
        """The --atoms-per-molecule flag should be accepted."""
        water = DATA_DIR / "water3.xyz"
        if not water.exists():
            pytest.skip("water3.xyz data not available")
        output = tmp_path / "water3.json"
        rc = main([
            "single",
            "--input", str(water),
            "--output", str(output),
            "--n-fragments", "2",
            "--atoms-per-molecule", "3",
        ])
        assert rc == 0

    def test_single_no_validate_water(self, tmp_path):
        """The --no-validate-water flag should be accepted."""
        water = DATA_DIR / "water3.xyz"
        if not water.exists():
            pytest.skip("water3.xyz data not available")
        output = tmp_path / "water3.json"
        rc = main([
            "single",
            "--input", str(water),
            "--output", str(output),
            "--n-fragments", "2",
            "--no-validate-water",
        ])
        assert rc == 0


class TestBatchCommand:
    """Tests for the 'batch' subcommand."""

    def test_batch_no_inputs_returns_1(self, tmp_path):
        """Batch with no input files should return 1."""
        water = DATA_DIR / "water3.xyz"
        if not water.exists():
            pytest.skip("water3.xyz data not available")
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        result = main([
            "batch",
            "--reference", str(water),
            "--output-dir", str(out_dir),
            # No --input-dir and no --inputs → empty
        ])
        assert result == 1

    def test_batch_duplicate_basenames(self, tmp_path):
        """Batch with duplicate basenames should disambiguate output names."""
        water = DATA_DIR / "water3.xyz"
        if not water.exists():
            pytest.skip("water3.xyz data not available")

        # Create two dirs with same-named files
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        shutil.copy(water, dir_a / "water.xyz")
        shutil.copy(water, dir_b / "water.xyz")

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        main([
            "batch",
            "--reference", str(water),
            "--output-dir", str(out_dir),
            "--inputs", str(dir_a / "water.xyz"), str(dir_b / "water.xyz"),
            "--n-fragments", "2",
            "--force",
        ])
        # Should produce two distinct output files
        outputs = list(out_dir.glob("*.json"))
        assert len(outputs) == 2


class TestInfoCommand:
    """Tests for the 'info' subcommand."""

    def test_info_returns_0(self):
        """The info command should succeed."""
        rc = main(["info"])
        assert rc == 0


class TestVersionFlag:
    """Tests for the --version flag."""

    def test_version_returns_0(self):
        """--version should succeed."""
        rc = main(["--version"])
        assert rc == 0
