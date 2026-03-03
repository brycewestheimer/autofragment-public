# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip file format tests."""
from pathlib import Path

import numpy as np
import pytest

from autofragment import Atom, ChemicalSystem, Fragment, io


@pytest.fixture
def benzene_system():
    # Simple benzene-like system
    atoms = [
        Atom("C", [0.0, 1.4, 0.0]),
        Atom("C", [1.2, 0.7, 0.0]),
        Atom("C", [1.2, -0.7, 0.0]),
        Atom("C", [0.0, -1.4, 0.0]),
        Atom("C", [-1.2, -0.7, 0.0]),
        Atom("C", [-1.2, 0.7, 0.0])
    ]
    return ChemicalSystem(atoms=atoms)

@pytest.fixture
def protein_system():
    # Load from data file
    data_path = Path(__file__).parent.parent / "data/protein.pdb"
    return io.read_pdb(str(data_path))


class TestRoundTrip:
    """Test read/write round-trips."""

    def test_xyz_roundtrip(self, benzene_system, tmp_path):
        """Test XYZ write then read."""
        # Use io.write_xyz directly
        system = benzene_system

        xyz_file = tmp_path / "test_roundtrip.xyz"

        # Write
        io.write_xyz(str(xyz_file), system)

        print(f"\nDEBUG: Content of {xyz_file}:")
        print(xyz_file.read_text())

        # Read back
        # io.read_xyz returns ChemicalSystem
        # Since benzene is not water (3 atoms), we set atoms_per_molecule=6
        # and disable water validation
        reloaded = io.read_xyz(
            str(xyz_file),
            validate_water=False,
            atoms_per_molecule=6
        )

        # Compare
        reloaded_mols = reloaded.to_molecules(require_metadata=True)
        assert len(reloaded_mols) == 1
        reloaded_atoms = reloaded_mols[0]
        assert len(reloaded_atoms) == len(benzene_system.atoms)

        for orig, new in zip(benzene_system.atoms, reloaded_atoms):
            assert orig.symbol == new.symbol
            assert np.allclose(orig.coords, new.coords, atol=1e-5)

    def test_pdb_roundtrip(self, protein_system, tmp_path):
        """Test PDB write then read."""
        # Convert system to one fragment to write it out using write_pdb_fragments
        frag = Fragment.from_molecules([protein_system.atoms], fragment_id="ALL")

        pdb_file = tmp_path / "test_roundtrip.pdb"

        # Write
        io.write_pdb_fragments([frag], str(pdb_file))

        # Read back
        reloaded = io.read_pdb(str(pdb_file))

        assert reloaded.n_atoms == protein_system.n_atoms
        # Verify coordinates/element of first atom
        assert reloaded.atoms[0].symbol == protein_system.atoms[0].symbol
        assert np.allclose(reloaded.atoms[0].coords, protein_system.atoms[0].coords, atol=1e-3)
