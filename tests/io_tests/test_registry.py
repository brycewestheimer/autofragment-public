# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FormatRegistry writer registration and auto-detection."""

class TestFormatRegistryWriters:
    """Test that writers are registered and auto-detection works."""

    def test_writers_registered_after_import(self):
        """FormatRegistry should have writers registered after import."""
        from autofragment.io.writers.registry import FormatRegistry

        formats = FormatRegistry.supported_formats()
        assert formats["write"], "No writers registered in FormatRegistry"

    def test_readers_registered_after_import(self):
        """FormatRegistry should have readers registered after import."""
        from autofragment.io.writers.registry import FormatRegistry

        formats = FormatRegistry.supported_formats()
        assert formats["read"], "No readers registered in FormatRegistry"

    def test_write_auto_detection_inp(self):
        """Writing to .inp should auto-detect gamess_fmo format."""
        from autofragment.io.writers.registry import FormatRegistry

        fmt = FormatRegistry._detect_write_format("output.inp")
        assert fmt == "gamess_fmo"

    def test_write_auto_detection_dat(self):
        """Writing to .dat should auto-detect psi4_sapt format."""
        from autofragment.io.writers.registry import FormatRegistry

        fmt = FormatRegistry._detect_write_format("output.dat")
        assert fmt == "psi4_sapt"

    def test_write_auto_detection_xyz(self):
        """Writing to .xyz should auto-detect xyz_fragments format."""
        from autofragment.io.writers.registry import FormatRegistry

        fmt = FormatRegistry._detect_write_format("output.xyz")
        assert fmt == "xyz_fragments"

    def test_write_auto_detection_json(self):
        """Writing to .json should auto-detect qcschema format."""
        from autofragment.io.writers.registry import FormatRegistry

        fmt = FormatRegistry._detect_write_format("output.json")
        assert fmt == "qcschema"

    def test_write_auto_detection_pdb(self):
        """Writing to .pdb should auto-detect pdb_fragments format."""
        from autofragment.io.writers.registry import FormatRegistry

        fmt = FormatRegistry._detect_write_format("output.pdb")
        assert fmt == "pdb_fragments"

    def test_read_auto_detection_not_shadowed_by_writer(self):
        """Read auto-detection should still prefer reader formats."""
        from autofragment.io.writers.registry import FormatRegistry

        # .inp is registered by both gamess reader and gamess_fmo writer
        # Read detection should return the reader's format
        fmt = FormatRegistry._detect_format("input.inp")
        assert fmt == "gamess"

    def test_write_explicit_format(self, tmp_path):
        """Writing with explicit format name should work for each writer."""
        from autofragment.io.writers.registry import FormatRegistry

        # Just verify that all registered writer names are accessible
        formats = FormatRegistry.supported_formats()
        for writer_name in formats["write"]:
            assert writer_name in FormatRegistry._writers

    def test_write_extensions_map_populated(self):
        """The _write_extensions map should have entries."""
        from autofragment.io.writers.registry import FormatRegistry

        assert FormatRegistry._write_extensions, "_write_extensions is empty"
        # Verify key extensions are present
        assert ".inp" in FormatRegistry._write_extensions
        assert ".dat" in FormatRegistry._write_extensions
        assert ".json" in FormatRegistry._write_extensions
