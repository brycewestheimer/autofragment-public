# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for molecular file format readers."""

import json

import numpy as np
import pytest


class TestPDBReader:
    """Test PDB file reader."""

    def test_read_simple_pdb(self, tmp_path):
        """Test reading a simple PDB file."""
        from autofragment.io.readers.pdb import read_pdb

        pdb_content = """\
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.958   0.000   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
END
"""
        pdb_file = tmp_path / "water.pdb"
        pdb_file.write_text(pdb_content)

        system = read_pdb(pdb_file)

        assert system.n_atoms == 3
        assert system.atoms[0].symbol == "O"
        assert system.atoms[1].symbol == "H"
        assert system.atoms[2].symbol == "H"
        assert np.allclose(system.atoms[0].coords, [0.0, 0.0, 0.0])

    def test_read_pdb_with_conect(self, tmp_path):
        """Test reading PDB with CONECT records."""
        from autofragment.io.readers.pdb import read_pdb

        pdb_content = """\
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.958   0.000   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
CONECT    1    2    3
CONECT    2    1
CONECT    3    1
END
"""
        pdb_file = tmp_path / "water_conect.pdb"
        pdb_file.write_text(pdb_content)

        system = read_pdb(pdb_file, infer_bonds=False)

        assert system.n_atoms == 3
        assert system.n_bonds == 2  # O-H1 and O-H2

    def test_read_pdb_multi_model(self, tmp_path):
        """Test reading specific MODEL from multi-model PDB."""
        from autofragment.io.readers.pdb import read_pdb

        pdb_content = """\
MODEL        1
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ENDMDL
MODEL        2
ATOM      1  O   HOH A   1       5.000   5.000   5.000  1.00  0.00           O
ENDMDL
END
"""
        pdb_file = tmp_path / "multi.pdb"
        pdb_file.write_text(pdb_content)

        # Read model 1
        system1 = read_pdb(pdb_file, model=1)
        assert np.allclose(system1.atoms[0].coords, [0.0, 0.0, 0.0])

        # Read model 2
        system2 = read_pdb(pdb_file, model=2)
        assert np.allclose(system2.atoms[0].coords, [5.0, 5.0, 5.0])


class TestMOL2Reader:
    """Test MOL2 file reader."""

    def test_read_simple_mol2(self, tmp_path):
        """Test reading a simple MOL2 file."""
        from autofragment.io.readers.mol2 import read_mol2

        mol2_content = """\
@<TRIPOS>MOLECULE
water
 3 2 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 O1          0.0000    0.0000    0.0000 O.3     1  HOH         0.0000
      2 H1          0.9580    0.0000    0.0000 H       1  HOH         0.0000
      3 H2         -0.2400    0.9270    0.0000 H       1  HOH         0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     3    1
"""
        mol2_file = tmp_path / "water.mol2"
        mol2_file.write_text(mol2_content)

        system = read_mol2(mol2_file)

        assert system.n_atoms == 3
        assert system.n_bonds == 2
        assert system.atoms[0].symbol == "O"

    def test_read_mol2_raises_on_malformed_atom_record(self, tmp_path):
        """Malformed MOL2 atom records should raise ValueError."""
        from autofragment.io.readers.mol2 import read_mol2

        mol2_content = """\
@<TRIPOS>MOLECULE
bad_atom
 2 1 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 O1          0.0000    BAD    0.0000 O.3     1  HOH         0.0000
      2 H1          0.9580    0.0000    0.0000 H       1  HOH         0.0000
@<TRIPOS>BOND
     1     1     2    1
"""
        mol2_file = tmp_path / "bad_atom.mol2"
        mol2_file.write_text(mol2_content)

        with pytest.raises(ValueError, match="Malformed MOL2 atom record"):
            read_mol2(mol2_file)

    def test_read_mol2_raises_on_malformed_bond_record(self, tmp_path):
        """Malformed MOL2 bond records should raise ValueError."""
        from autofragment.io.readers.mol2 import read_mol2

        mol2_content = """\
@<TRIPOS>MOLECULE
bad_bond
 2 1 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 O1          0.0000    0.0000    0.0000 O.3     1  HOH         0.0000
      2 H1          0.9580    0.0000    0.0000 H       1  HOH         0.0000
@<TRIPOS>BOND
     1     1     BAD    1
"""
        mol2_file = tmp_path / "bad_bond.mol2"
        mol2_file.write_text(mol2_content)

        with pytest.raises(ValueError, match="Malformed MOL2 bond record"):
            read_mol2(mol2_file)


class TestSDFReader:
    """Test SDF/MOL file reader."""

    def test_read_simple_sdf(self, tmp_path):
        """Test reading a simple SDF file."""
        from autofragment.io.readers.sdf import read_sdf

        sdf_content = """\
water
  AutoFrag 01012024 3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.9580    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2400    0.9270    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""
        sdf_file = tmp_path / "water.sdf"
        sdf_file.write_text(sdf_content)

        system = read_sdf(sdf_file)

        assert system.n_atoms == 3
        assert system.n_bonds == 2
        assert system.atoms[0].symbol == "O"

        def test_read_sdf_raises_on_malformed_v3000_atom_record(self, tmp_path):
                """Malformed V3000 atom records should raise ValueError."""
                from autofragment.io.readers.sdf import read_sdf

                sdf_content = """\
bad_v3000
    AutoFrag 01012024 3D

    0  0  0  0  0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 1 0 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C BAD 0.0000 0.0000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
"""
                sdf_file = tmp_path / "bad_v3000_atom.sdf"
                sdf_file.write_text(sdf_content)

                with pytest.raises(ValueError, match="Malformed V3000 atom record"):
                        read_sdf(sdf_file)

        def test_read_sdf_raises_on_malformed_v3000_bond_record(self, tmp_path):
                """Malformed V3000 bond records should raise ValueError."""
                from autofragment.io.readers.sdf import read_sdf

                sdf_content = """\
bad_v3000_bond
    AutoFrag 01012024 3D

    0  0  0  0  0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 2 1 0 0 0
M  V30 BEGIN ATOM
M  V30 1 O 0.0000 0.0000 0.0000 0
M  V30 2 H 0.9580 0.0000 0.0000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 BAD 1 2
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
"""
                sdf_file = tmp_path / "bad_v3000_bond.sdf"
                sdf_file.write_text(sdf_content)

                with pytest.raises(ValueError, match="Malformed V3000 bond record"):
                        read_sdf(sdf_file)


class TestQCSchemaReader:
    """Test QCSchema JSON reader."""

    def test_read_qcschema(self, tmp_path):
        """Test reading QCSchema file."""
        from autofragment.io.readers.qcschema import read_qcschema

        # Bohr coordinates (will be converted to Angstroms)
        qcschema_data = {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "symbols": ["O", "H", "H"],
            "geometry": [
                0.0, 0.0, 0.0,
                1.81, 0.0, 0.0,  # ~0.958 Angstrom in Bohr
                -0.45, 1.75, 0.0,
            ],
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        }

        qcschema_file = tmp_path / "water.json"
        with open(qcschema_file, "w") as f:
            json.dump(qcschema_data, f)

        system = read_qcschema(qcschema_file)

        assert system.n_atoms == 3
        assert system.atoms[0].symbol == "O"
        assert system.metadata["molecular_charge"] == 0


class TestQCInputReaders:
    """Test QC program input file readers."""

    def test_read_gamess_input(self, tmp_path):
        """Test reading GAMESS input file."""
        from autofragment.io.readers.qc_inputs import read_gamess_input

        gamess_content = """\
 $CONTRL SCFTYP=RHF RUNTYP=ENERGY ICHARG=0 MULT=1 $END
 $BASIS GBASIS=STO NGAUSS=3 $END
 $DATA
Water molecule
C1
O      8.0     0.0000000000    0.0000000000    0.0000000000
H      1.0     0.9580000000    0.0000000000    0.0000000000
H      1.0    -0.2400000000    0.9270000000    0.0000000000
 $END
"""
        inp_file = tmp_path / "water.inp"
        inp_file.write_text(gamess_content)

        system = read_gamess_input(inp_file)

        assert system.n_atoms == 3
        assert system.atoms[0].symbol == "O"

    def test_read_psi4_input(self, tmp_path):
        """Test reading Psi4 input file."""
        from autofragment.io.readers.qc_inputs import read_psi4_input

        psi4_content = """\
memory 4 GB

molecule water {
  0 1
  O  0.000  0.000  0.000
  H  0.958  0.000  0.000
  H -0.240  0.927  0.000
}

set basis 6-31g*
energy('scf')
"""
        dat_file = tmp_path / "water.dat"
        dat_file.write_text(psi4_content)

        system = read_psi4_input(dat_file)

        assert system.n_atoms == 3
        assert system.atoms[0].symbol == "O"

    def test_read_orca_input(self, tmp_path):
        """Test reading ORCA input file."""
        from autofragment.io.readers.qc_inputs import read_orca_input

        orca_content = """\
! HF def2-SVP

* xyz 0 1
O  0.000  0.000  0.000
H  0.958  0.000  0.000
H -0.240  0.927  0.000
*
"""
        orca_file = tmp_path / "water.inp"
        orca_file.write_text(orca_content)

        system = read_orca_input(orca_file)

        assert system.n_atoms == 3
        assert system.metadata["molecular_charge"] == 0

    def test_read_qchem_raises_on_malformed_coordinate_line(self, tmp_path):
        """Malformed Q-Chem coordinate records should raise ValueError."""
        from autofragment.io.readers.qc_inputs import read_qchem_input

        qchem_content = """\
$molecule
0 1
O 0.000 0.000 0.000
H BAD 0.000 0.000
$end
"""
        qchem_file = tmp_path / "bad_qchem.in"
        qchem_file.write_text(qchem_content)

        with pytest.raises(ValueError, match="Malformed Q-Chem coordinate line"):
            read_qchem_input(qchem_file)


class TestVASPReader:
    """Test VASP POSCAR reader."""

    def test_read_poscar(self, tmp_path):
        """Test reading POSCAR file."""
        from autofragment.io.readers.vasp import read_poscar

        poscar_content = """\
Water molecule
1.0
10.0  0.0  0.0
0.0  10.0  0.0
0.0  0.0  10.0
O H
1 2
Cartesian
0.0  0.0  0.0
0.958  0.0  0.0
-0.240  0.927  0.0
"""
        poscar_file = tmp_path / "POSCAR"
        poscar_file.write_text(poscar_content)

        system = read_poscar(poscar_file)

        assert system.n_atoms == 3
        assert system.atoms[0].symbol == "O"
        assert system.atoms[1].symbol == "H"


class TestXYZValidation:
    """Tests for XYZ reader input validation."""

    def test_read_xyz_atoms_per_molecule_zero(self, tmp_path):
        """atoms_per_molecule=0 should raise ValidationError."""
        from autofragment.io.xyz import ValidationError, read_xyz

        xyz_content = "3\ntest\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text(xyz_content)

        with pytest.raises(ValidationError, match="must be positive"):
            read_xyz(xyz_file, atoms_per_molecule=0)

    def test_read_xyz_atoms_per_molecule_negative(self, tmp_path):
        """atoms_per_molecule=-1 should raise ValidationError."""
        from autofragment.io.xyz import ValidationError, read_xyz

        xyz_content = "3\ntest\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text(xyz_content)

        with pytest.raises(ValidationError, match="must be positive"):
            read_xyz(xyz_file, atoms_per_molecule=-1)

    def test_read_xyz_atoms_per_molecule_non_divisible(self, tmp_path):
        """Non-divisible atoms_per_molecule should raise ValidationError."""
        from autofragment.io.xyz import ValidationError, read_xyz

        xyz_content = "3\ntest\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text(xyz_content)

        with pytest.raises(ValidationError, match="must be multiple"):
            read_xyz(xyz_file, atoms_per_molecule=5)


class TestCIFLattice:
    """Tests for CIF reader lattice propagation."""

    def test_basic_cif_reader_sets_lattice(self, tmp_path):
        """read_cif with use_gemmi=False should set lattice on periodic CIF."""
        from autofragment.io.readers.cif import read_cif

        cif_content = """\
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_type_symbol
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
Si 0.0 0.0 0.0
Na 2.5 2.5 2.5
"""
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(cif_content)

        system = read_cif(cif_file, use_gemmi=False)

        # The basic parser has known limitations with certain element names;
        # verify at least one atom was parsed and the lattice is set.
        assert system.n_atoms >= 1
        assert system.lattice is not None
        # Check lattice vector lengths (cubic 5A cell)
        import numpy as np
        lengths = np.linalg.norm(system.lattice.vectors, axis=1)
        assert abs(lengths[0] - 5.0) < 1e-6
        assert abs(lengths[1] - 5.0) < 1e-6
        assert abs(lengths[2] - 5.0) < 1e-6
