# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for autofragment.io module."""

import tempfile
from pathlib import Path

import pytest

from autofragment.core.types import Atom
from autofragment.io.xyz import ValidationError, read_xyz, read_xyz_molecules, write_xyz


class TestReadXYZ:
    """Tests for read_xyz function."""

    def test_read_water16(self, water16_path):
        """Test reading water16 test file."""
        system = read_xyz(water16_path)
        molecules = system.to_molecules(require_metadata=True)
        assert len(molecules) == 16
        for mol in molecules:
            assert len(mol) == 3
            assert mol[0].symbol.upper() == "O"
            assert mol[1].symbol.upper() == "H"
            assert mol[2].symbol.upper() == "H"

    def test_read_xyz_angstrom(self, water16_path):
        """Test reading XYZ in angstrom units."""
        system = read_xyz(water16_path, xyz_units="angstrom")
        molecules = system.to_molecules(require_metadata=True)
        assert len(molecules) == 16

    def test_read_xyz_not_found(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(ValidationError):
            read_xyz("/nonexistent/file.xyz")

    def test_read_xyz_bad_format(self):
        """Test reading malformed XYZ file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("not a number\n")
            f.write("comment\n")
            f.write("O 0 0 0\n")
            path = f.name

        with pytest.raises(ValidationError):
            read_xyz(path)

        Path(path).unlink()


class TestWriteXYZ:
    """Tests for write_xyz function."""

    def test_write_and_read(self, water16_path):
        """Test writing and reading back XYZ file."""
        system = read_xyz(water16_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.xyz"
            write_xyz(path, system, comment="test comment")

            assert path.exists()

            # Read back and verify
            loaded = read_xyz(path)
            assert loaded.n_atoms == system.n_atoms

    def test_write_single_molecule(self):
        """Test writing a single molecule."""
        molecules = [
            [
                Atom("O", [0.0, 0.0, 0.0]),
                Atom("H", [1.0, 0.0, 0.0]),
                Atom("H", [0.0, 1.0, 0.0]),
            ]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "water.xyz"
            write_xyz(path, molecules)

            content = path.read_text()
            lines = content.strip().split("\n")
            assert lines[0] == "3"
            assert lines[1] == ""  # empty comment

    def test_read_xyz_molecules_helper(self, water16_path):
        """Test explicit molecule helper for XYZ."""
        molecules = read_xyz_molecules(water16_path)
        assert len(molecules) == 16
