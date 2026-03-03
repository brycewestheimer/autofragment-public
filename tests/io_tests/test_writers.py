# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for molecular file format writers."""

import json

import numpy as np
import pytest

from autofragment.core.types import Atom, ChemicalSystem, Fragment


@pytest.fixture
def water_fragment():
    """Create a simple water fragment."""
    return Fragment(
        id="W1",
        symbols=["O", "H", "H"],
        geometry=[0.0, 0.0, 0.0, 0.958, 0.0, 0.0, -0.240, 0.927, 0.0],
        molecular_charge=0,
        molecular_multiplicity=1,
    )


@pytest.fixture
def water_dimer_fragments():
    """Create two water fragments."""
    return [
        Fragment(
            id="W1",
            symbols=["O", "H", "H"],
            geometry=[0.0, 0.0, 0.0, 0.958, 0.0, 0.0, -0.240, 0.927, 0.0],
            molecular_charge=0,
            molecular_multiplicity=1,
        ),
        Fragment(
            id="W2",
            symbols=["O", "H", "H"],
            geometry=[3.0, 0.0, 0.0, 3.958, 0.0, 0.0, 2.760, 0.927, 0.0],
            molecular_charge=0,
            molecular_multiplicity=1,
        ),
    ]


class TestGAMESSWriter:
    """Test GAMESS FMO writer."""

    def test_write_gamess_fmo(self, tmp_path, water_dimer_fragments):
        """Test writing GAMESS FMO input."""
        from autofragment.io.writers.gamess import write_gamess_fmo

        output_file = tmp_path / "dimer.inp"
        write_gamess_fmo(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        # Check for key sections
        assert "$CONTRL" in content
        assert "$FMO" in content
        assert "$DATA" in content
        assert "NFRAG=2" in content

    def test_write_gamess_fmo_basis(self, tmp_path, water_fragment):
        """Test GAMESS writer with different basis sets."""
        from autofragment.io.writers.gamess import write_gamess_fmo

        output_file = tmp_path / "water.inp"
        write_gamess_fmo([water_fragment], output_file, basis="cc-pVDZ")

        content = output_file.read_text()
        assert "$BASIS" in content


class TestPsi4Writer:
    """Test Psi4 writer."""

    def test_write_psi4_sapt(self, tmp_path, water_dimer_fragments):
        """Test writing Psi4 SAPT input."""
        from autofragment.io.writers.psi4 import write_psi4_sapt

        output_file = tmp_path / "dimer.dat"
        write_psi4_sapt(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        # Check for key elements
        assert "molecule" in content
        assert "sapt" in content.lower()
        assert "--" in content  # Fragment separator

    def test_write_psi4_fragment(self, tmp_path, water_fragment):
        """Test writing Psi4 fragment input."""
        from autofragment.io.writers.psi4 import write_psi4_fragment

        output_file = tmp_path / "water.dat"
        write_psi4_fragment([water_fragment], output_file, method="hf")

        content = output_file.read_text()
        assert "molecule" in content
        assert "0 1" in content  # Charge and multiplicity


class TestQChemWriter:
    """Test Q-Chem writer."""

    def test_write_qchem_efp(self, tmp_path, water_dimer_fragments):
        """Test writing Q-Chem EFP input."""
        from autofragment.io.writers.qchem import write_qchem_efp

        output_file = tmp_path / "dimer.in"
        write_qchem_efp(water_dimer_fragments, output_file, qm_fragment_indices=[0])

        assert output_file.exists()
        content = output_file.read_text()

        assert "$molecule" in content
        assert "$rem" in content


class TestORCAWriter:
    """Test ORCA writer."""

    def test_write_orca_fragment(self, tmp_path, water_dimer_fragments):
        """Test writing ORCA input."""
        from autofragment.io.writers.orca import write_orca_fragment

        output_file = tmp_path / "dimer.inp"
        write_orca_fragment(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        assert "!" in content  # Keywords line
        assert "* xyz" in content

    def test_write_orca_multijob(self, tmp_path, water_dimer_fragments):
        """Test writing ORCA multi-job input."""
        from autofragment.io.writers.orca import write_orca_multijob

        output_file = tmp_path / "dimer.inp"
        write_orca_multijob(water_dimer_fragments, output_file)

        content = output_file.read_text()
        assert "$new_job" in content


class TestQCSchemaWriter:
    """Test QCSchema writer."""

    def test_write_qcschema(self, tmp_path, water_fragment):
        """Test writing QCSchema JSON."""
        from autofragment.io.writers.qcschema_writer import write_qcschema

        # Create a simple system
        atoms = [
            Atom(symbol="O", coords=np.array([0.0, 0.0, 0.0])),
            Atom(symbol="H", coords=np.array([0.958, 0.0, 0.0])),
            Atom(symbol="H", coords=np.array([-0.240, 0.927, 0.0])),
        ]
        system = ChemicalSystem(atoms=atoms)

        output_file = tmp_path / "molecule.json"
        write_qcschema(system, output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["schema_name"] == "qcschema_molecule"
        assert len(data["symbols"]) == 3
        assert "geometry" in data

    def test_system_to_qcschema_conversion(self):
        """Test Angstrom to Bohr conversion."""
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        atoms = [Atom(symbol="H", coords=np.array([0.0, 0.0, 0.0]))]
        system = ChemicalSystem(atoms=atoms)

        result = system_to_qcschema(system)

        # Geometry should be in Bohr (0 Angstrom = 0 Bohr)
        assert result["geometry"] == [0.0, 0.0, 0.0]


class TestXYZWriter:
    """Test XYZ writer with fragment markers."""

    def test_write_xyz_fragments(self, tmp_path, water_dimer_fragments):
        """Test writing XYZ with fragment markers."""
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        output_file = tmp_path / "dimer.xyz"
        write_xyz_fragments(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        lines = content.strip().split("\n")
        assert lines[0] == "6"  # 6 atoms total
        assert "Fragments" in lines[1]  # Comment line


class TestPDBWriter:
    """Test PDB writer with fragment encoding."""

    def test_write_pdb_chains(self, tmp_path, water_dimer_fragments):
        """Test writing PDB with fragments as chains."""
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        output_file = tmp_path / "dimer.pdb"
        write_pdb_fragments(water_dimer_fragments, output_file, mode="chains")

        assert output_file.exists()
        content = output_file.read_text()

        assert "ATOM" in content
        assert " A " in content  # Chain A
        assert " B " in content  # Chain B
        assert "END" in content

    def test_write_pdb_models(self, tmp_path, water_dimer_fragments):
        """Test writing PDB with fragments as models."""
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        output_file = tmp_path / "dimer.pdb"
        write_pdb_fragments(water_dimer_fragments, output_file, mode="models")

        content = output_file.read_text()
        assert "MODEL" in content
        assert "ENDMDL" in content


class TestFormatRegistry:
    """Test format registry system."""

    def test_registry_supported_formats(self):
        """Test listing supported formats."""
        from autofragment.io.writers.registry import FormatRegistry

        formats = FormatRegistry.supported_formats()

        assert "read" in formats
        assert "write" in formats

    def test_registry_custom_format(self):
        """Test registering a custom format."""
        from autofragment.io.writers.registry import FormatRegistry

        def custom_reader(filepath):
            return ChemicalSystem()

        FormatRegistry.register_reader(
            "custom",
            custom_reader,
            extensions=[".custom"],
            description="Custom test format"
        )

        formats = FormatRegistry.supported_formats()
        assert "custom" in formats["read"]

        # Clean up
        FormatRegistry.unregister_reader("custom")
