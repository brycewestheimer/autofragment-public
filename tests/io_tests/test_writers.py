# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for molecular file format writers."""

import json
import re

import numpy as np
import pytest

from autofragment.core.types import Atom, ChemicalSystem, Fragment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def three_fragment_system():
    """Three water fragments (9 atoms). Tests INDAT wrapping, multi-fragment handling."""
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
        Fragment(
            id="W3",
            symbols=["O", "H", "H"],
            geometry=[6.0, 0.0, 0.0, 6.958, 0.0, 0.0, 5.760, 0.927, 0.0],
            molecular_charge=0,
            molecular_multiplicity=1,
        ),
    ]


@pytest.fixture
def bonded_dipeptide_fragments():
    """Two glycine-like residue fragments connected by a C-N peptide bond.

    Fragment "RES1": [N, C, C, O, H, H, H] -- C at local index 2 is bond endpoint
    Fragment "RES2": [N, C, C, O, H, H, H] -- N at local index 0 is bond endpoint
    """
    frag1 = Fragment(
        id="RES1",
        symbols=["N", "C", "C", "O", "H", "H", "H"],
        geometry=[
            0.0, 0.0, 0.0,    # N
            1.47, 0.0, 0.0,   # CA
            2.40, 1.20, 0.0,  # C (carbonyl - bond endpoint)
            2.10, 2.40, 0.0,  # O
            0.0, -1.0, 0.0,   # H
            1.47, 0.0, 1.09,  # H
            1.47, 0.0, -1.09, # H
        ],
        molecular_charge=0,
        molecular_multiplicity=1,
    )
    frag2 = Fragment(
        id="RES2",
        symbols=["N", "C", "C", "O", "H", "H", "H"],
        geometry=[
            3.80, 1.00, 0.0,  # N (bond endpoint)
            4.90, 2.00, 0.0,  # CA
            6.00, 1.20, 0.0,  # C
            6.50, 2.30, 0.0,  # O
            3.80, 0.0, 0.0,   # H
            4.90, 2.00, 1.09, # H
            4.90, 2.00, -1.09,# H
        ],
        molecular_charge=0,
        molecular_multiplicity=1,
    )
    interfragment_bonds = [
        {
            "fragment1_id": "RES1",
            "atom1_index": 2,
            "fragment2_id": "RES2",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {"type": "peptide"},
        }
    ]
    return frag1, frag2, interfragment_bonds


@pytest.fixture
def water_dimer_system():
    """A ChemicalSystem with 6 atoms (two waters), for QCSchema writer tests."""
    atoms = [
        Atom(symbol="O", coords=np.array([0.0, 0.0, 0.0])),
        Atom(symbol="H", coords=np.array([0.958, 0.0, 0.0])),
        Atom(symbol="H", coords=np.array([-0.240, 0.927, 0.0])),
        Atom(symbol="O", coords=np.array([3.0, 0.0, 0.0])),
        Atom(symbol="H", coords=np.array([3.958, 0.0, 0.0])),
        Atom(symbol="H", coords=np.array([2.760, 0.927, 0.0])),
    ]
    return ChemicalSystem(atoms=atoms)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read(tmp_path, name):
    return (tmp_path / name).read_text()


# ===========================================================================
# GAMESS
# ===========================================================================

class TestGAMESSWriter:
    """Test GAMESS FMO writer."""

    def test_write_gamess_fmo(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        output_file = tmp_path / "dimer.inp"
        write_gamess_fmo(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        assert "$CONTRL" in content
        assert "$FMO" in content
        assert "$DATA" in content
        assert "NFRAG=2" in content

    def test_write_gamess_fmo_basis(self, tmp_path, water_fragment):
        from autofragment.io.writers.gamess import write_gamess_fmo

        output_file = tmp_path / "water.inp"
        write_gamess_fmo([water_fragment], output_file, basis="cc-pVDZ")

        content = output_file.read_text()
        assert "$BASIS" in content


class TestGAMESSFormatCorrectness:
    """Thorough format-correctness tests for GAMESS writers."""

    def test_group_delimiters(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # Every group starts with " $NAME" (space-dollar) and ends with " $END"
        groups = re.findall(r" \$(\w+)", content)
        for group_name in ["CONTRL", "SYSTEM", "BASIS", "FMO", "GDDI", "DATA"]:
            assert group_name in groups, f"Missing group ${group_name}"

        # Each group that opens should close
        end_count = content.count(" $END")
        assert end_count >= 6

    def test_contrl_keywords_default(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "SCFTYP=RHF" in content
        assert "RUNTYP=ENERGY" in content
        assert "COORD=UNIQUE" in content

    def test_contrl_custom_method_runtype(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(
            water_dimer_fragments, tmp_path / "t.inp", method="UHF", runtype="gradient"
        )
        content = _read(tmp_path, "t.inp")

        assert "SCFTYP=UHF" in content
        assert "RUNTYP=GRADIENT" in content

    def test_system_memory(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp", memory=1000)
        content = _read(tmp_path, "t.inp")

        assert "MWORDS=1000" in content

    def test_system_memory_default(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "MWORDS=500" in content

    def test_basis_631g_star(self, tmp_path, water_fragment):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo([water_fragment], tmp_path / "t.inp", basis="6-31G*")
        content = _read(tmp_path, "t.inp")

        assert "N31 NGAUSS=6 NDFUNC=1" in content

    def test_basis_cc_pvdz(self, tmp_path, water_fragment):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo([water_fragment], tmp_path / "t.inp", basis="cc-pVDZ")
        content = _read(tmp_path, "t.inp")

        assert "CCD" in content

    def test_basis_unknown_fallback(self, tmp_path, water_fragment):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo([water_fragment], tmp_path / "t.inp", basis="exotic-basis")
        content = _read(tmp_path, "t.inp")

        assert "GBASIS=EXOTIC-BASIS" in content

    def test_fmo_nfrag_nbody(self, tmp_path, three_fragment_system):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(three_fragment_system, tmp_path / "t.inp", nbody=3)
        content = _read(tmp_path, "t.inp")

        assert "NFRAG=3" in content
        assert "NBODY=3" in content

    def test_icharg_mult_arrays(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "ICHARG(1)=0,0" in content
        assert "MULT(1)=1,1" in content

    def test_indat_values_water_dimer(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # 1-indexed fragment assignments: first 3 atoms -> frag 1, next 3 -> frag 2
        assert "INDAT(1)=1,1,1,2,2,2" in content

    def test_indat_wrapping_many_atoms(self, tmp_path):
        """INDAT wraps at 10 values per line with 13-space continuation."""
        from autofragment.io.writers.gamess import write_gamess_fmo

        # 4 water fragments = 12 atoms, should wrap
        frags = [
            Fragment(
                id=f"W{i}",
                symbols=["O", "H", "H"],
                geometry=[i * 3.0, 0.0, 0.0, i * 3.0 + 0.958, 0.0, 0.0,
                          i * 3.0 - 0.240, 0.927, 0.0],
            )
            for i in range(4)
        ]
        write_gamess_fmo(frags, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # First line has INDAT(1)= prefix + up to 10 values
        # Continuation lines have 13-space indent
        lines = content.split("\n")
        indat_lines = [l for l in lines if "INDAT" in l or (l.startswith("             ") and "," in l)]
        assert len(indat_lines) >= 2  # Should wrap

    def test_data_group_format(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # DATA group: title, C1, atoms
        assert "C1" in content

        # O should map to 8.0, H to 1.0
        assert "8.0" in content  # oxygen Z
        assert "1.0" in content  # hydrogen Z

    def test_data_atom_znuc_values(self, tmp_path, water_fragment):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo([water_fragment], tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # Find atom lines in DATA group (between $DATA and $END)
        in_data = False
        atom_lines = []
        for line in content.split("\n"):
            if "$DATA" in line:
                in_data = True
                continue
            if in_data and "$END" in line:
                break
            if in_data and line.strip() and line.strip() not in ("C1",) and "AutoFragment" not in line:
                atom_lines.append(line)

        # Should have 3 atom lines (O, H, H)
        assert len(atom_lines) == 3
        # First atom should be O with znuc 8.0
        assert "O" in atom_lines[0]
        assert "8.0" in atom_lines[0]

    def test_efmo_has_modgrd(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_efmo

        write_gamess_efmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "MODGRD=1" in content

    def test_efp_sections(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_efp

        write_gamess_efp(
            water_dimer_fragments, [0], tmp_path / "t.inp"
        )
        content = _read(tmp_path, "t.inp")

        assert "$EFRAG" in content
        assert "NFRAGS=1" in content  # 1 EFP fragment
        assert "FRAGNAME(1)=WATER" in content
        assert "$EFCOOR" in content
        assert "$DATA" in content

    def test_efp_coord_precision(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_efp

        write_gamess_efp(
            water_dimer_fragments, [0], tmp_path / "t.inp"
        )
        content = _read(tmp_path, "t.inp")

        # EFP coords use 12.8f precision
        # Check that a coordinate appears with ~8 decimal places
        coord_pattern = re.compile(r"\d+\.\d{8}")
        assert coord_pattern.search(content)

    def test_efp_qm_atoms_only_in_data(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_efp

        write_gamess_efp(
            water_dimer_fragments, [0], tmp_path / "t.inp"
        )
        content = _read(tmp_path, "t.inp")

        # DATA group should only have QM fragment atoms (3 atoms for W1)
        in_data = False
        atom_lines = []
        for line in content.split("\n"):
            if "$DATA" in line:
                in_data = True
                continue
            if in_data and "$END" in line:
                break
            if in_data and line.strip() and line.strip() != "C1" and "AutoFragment" not in line:
                atom_lines.append(line)

        assert len(atom_lines) == 3

    def test_extra_contrl_passthrough(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(
            water_dimer_fragments, tmp_path / "t.inp",
            extra_contrl={"ISPHER": "1"},
        )
        content = _read(tmp_path, "t.inp")

        assert "ISPHER=1" in content

    def test_extra_fmo_passthrough(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(
            water_dimer_fragments, tmp_path / "t.inp",
            extra_fmo={"RESPPC": "1"},
        )
        content = _read(tmp_path, "t.inp")

        assert "RESPPC=1" in content


# ===========================================================================
# Psi4
# ===========================================================================

class TestPsi4Writer:
    """Test Psi4 writer."""

    def test_write_psi4_sapt(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        output_file = tmp_path / "dimer.dat"
        write_psi4_sapt(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        assert "molecule" in content
        assert "sapt" in content.lower()
        assert "--" in content

    def test_write_psi4_fragment(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        output_file = tmp_path / "water.dat"
        write_psi4_fragment([water_fragment], output_file, method="hf")

        content = output_file.read_text()
        assert "molecule" in content
        assert "0 1" in content


class TestPsi4FormatCorrectness:
    """Thorough format-correctness tests for Psi4 writers."""

    def test_sapt_validation_too_few_frags(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        with pytest.raises(ValueError, match="at least 2"):
            write_psi4_sapt([water_fragment], tmp_path / "t.dat")

    def test_sapt_molecule_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        write_psi4_sapt(water_dimer_fragments, tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        assert "molecule complex {" in content
        assert content.count("--") == 1  # One separator for 2 fragments
        assert "}" in content

    def test_sapt_charge_mult_lines(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        write_psi4_sapt(water_dimer_fragments, tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        assert "  0 1" in content

    def test_sapt_set_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        write_psi4_sapt(water_dimer_fragments, tmp_path / "t.dat", basis="aug-cc-pvdz")
        content = _read(tmp_path, "t.dat")

        assert "set {" in content
        assert "basis aug-cc-pvdz" in content
        assert "freeze_core true" in content
        assert "scf_type df" in content

    def test_sapt_energy_call(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        write_psi4_sapt(water_dimer_fragments, tmp_path / "t.dat", method="sapt2+")
        content = _read(tmp_path, "t.dat")

        assert "energy('sapt2+')" in content

    def test_fragment_single_no_separator(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        write_psi4_fragment([water_fragment], tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        assert "--" not in content

    def test_fragment_multiple_has_separators(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        write_psi4_fragment(water_dimer_fragments, tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        assert "--" in content

    def test_fragment_runtype_energy(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        write_psi4_fragment([water_fragment], tmp_path / "t.dat", method="scf", runtype="energy")
        content = _read(tmp_path, "t.dat")

        assert "energy('scf')" in content

    def test_fragment_runtype_gradient(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        write_psi4_fragment([water_fragment], tmp_path / "t.dat", method="scf", runtype="gradient")
        content = _read(tmp_path, "t.dat")

        assert "gradient('scf')" in content

    def test_fragment_runtype_optimize(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fragment

        write_psi4_fragment([water_fragment], tmp_path / "t.dat", method="scf", runtype="optimize")
        content = _read(tmp_path, "t.dat")

        assert "optimize('scf')" in content

    def test_fsapt_energy_call(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_fsapt

        write_psi4_fsapt(water_dimer_fragments, tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        assert "energy('fisapt0')" in content
        assert "molecule dimer {" in content

    def test_fsapt_too_few_frags(self, tmp_path, water_fragment):
        from autofragment.io.writers.psi4 import write_psi4_fsapt

        with pytest.raises(ValueError, match="at least 2"):
            write_psi4_fsapt([water_fragment], tmp_path / "t.dat")

    def test_coord_format_precision(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.psi4 import write_psi4_sapt

        write_psi4_sapt(water_dimer_fragments, tmp_path / "t.dat")
        content = _read(tmp_path, "t.dat")

        # Coords should be at 15.10f precision
        coord_pattern = re.compile(r"\d+\.\d{10}")
        assert coord_pattern.search(content)


# ===========================================================================
# Q-Chem
# ===========================================================================

class TestQChemWriter:
    """Test Q-Chem writer."""

    def test_write_qchem_efp(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_efp

        output_file = tmp_path / "dimer.in"
        write_qchem_efp(water_dimer_fragments, output_file, qm_fragment_indices=[0])

        assert output_file.exists()
        content = output_file.read_text()

        assert "$molecule" in content
        assert "$rem" in content


class TestQChemFormatCorrectness:
    """Thorough format-correctness tests for Q-Chem writers."""

    def test_efp_section_delimiters(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_efp

        write_qchem_efp(
            water_dimer_fragments, tmp_path / "t.in", qm_fragment_indices=[0]
        )
        content = _read(tmp_path, "t.in")

        assert "$molecule" in content
        assert "$end" in content
        assert "$rem" in content
        assert "$comment" in content

    def test_efp_rem_keywords(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_efp

        write_qchem_efp(
            water_dimer_fragments, tmp_path / "t.in", qm_fragment_indices=[0]
        )
        content = _read(tmp_path, "t.in")

        assert "EFP = TRUE" in content
        assert "$efp_fragments" in content

    def test_efp_qm_atoms_only(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_efp

        write_qchem_efp(
            water_dimer_fragments, tmp_path / "t.in", qm_fragment_indices=[0]
        )
        content = _read(tmp_path, "t.in")

        # Between $molecule and $end, only QM atoms (3 atoms from frag 0)
        mol_match = re.search(r"\$molecule\n(.+?)\$end", content, re.DOTALL)
        assert mol_match
        mol_block = mol_match.group(1)
        # First line is charge/mult, rest are atoms
        mol_lines = [l for l in mol_block.strip().split("\n") if l.strip()]
        # charge/mult line + 3 atom lines
        assert len(mol_lines) == 4

    def test_xsapt_validation(self, tmp_path, water_fragment):
        from autofragment.io.writers.qchem import write_qchem_xsapt

        with pytest.raises(ValueError, match="at least 2"):
            write_qchem_xsapt([water_fragment], tmp_path / "t.in")

    def test_xsapt_rem_keywords(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_xsapt

        write_qchem_xsapt(water_dimer_fragments, tmp_path / "t.in")
        content = _read(tmp_path, "t.in")

        assert "XSAPT = TRUE" in content

    def test_xsapt_separators(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_xsapt

        write_qchem_xsapt(water_dimer_fragments, tmp_path / "t.in")
        content = _read(tmp_path, "t.in")

        assert "--" in content

    def test_fragmo_rem_keywords(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_fragmo

        write_qchem_fragmo(water_dimer_fragments, tmp_path / "t.in")
        content = _read(tmp_path, "t.in")

        assert "FRAGMO = TRUE" in content
        assert "NFRAGMO = 2" in content

    def test_fragmo_separators(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.qchem import write_qchem_fragmo

        write_qchem_fragmo(water_dimer_fragments, tmp_path / "t.in")
        content = _read(tmp_path, "t.in")

        assert "--" in content


# ===========================================================================
# NWChem
# ===========================================================================

class TestNWChemFormatCorrectness:
    """Format-correctness tests for NWChem writers."""

    def test_geometry_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw")
        content = _read(tmp_path, "t.nw")

        assert "geometry units angstroms noautoz" in content
        assert content.count("end") >= 2  # geometry end + basis end

    def test_basis_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw", basis="cc-pvdz")
        content = _read(tmp_path, "t.nw")

        assert "basis" in content
        assert "* library cc-pvdz" in content

    def test_charge_line(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw")
        content = _read(tmp_path, "t.nw")

        assert "charge 0" in content

    def test_method_dft(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw", method="dft")
        content = _read(tmp_path, "t.nw")

        assert "xc b3lyp" in content
        assert "task dft energy" in content

    def test_method_mp2(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw", method="mp2")
        content = _read(tmp_path, "t.nw")

        assert "freeze core" in content
        assert "task mp2 energy" in content

    def test_method_ccsd(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw", method="ccsd")
        content = _read(tmp_path, "t.nw")

        assert "freeze core" in content
        assert "task ccsd energy" in content

    def test_method_scf_no_extra_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_fragment

        write_nwchem_fragment(water_dimer_fragments, tmp_path / "t.nw", method="scf")
        content = _read(tmp_path, "t.nw")

        assert "task scf energy" in content
        assert "xc b3lyp" not in content
        assert "freeze core" not in content

    def test_bsse_validation(self, tmp_path, water_fragment):
        from autofragment.io.writers.nwchem import write_nwchem_bsse

        with pytest.raises(ValueError, match="at least 2"):
            write_nwchem_bsse([water_fragment], tmp_path / "t.nw")

    def test_bsse_multiple_tasks(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_bsse

        write_nwchem_bsse(water_dimer_fragments, tmp_path / "t.nw")
        content = _read(tmp_path, "t.nw")

        task_count = content.count("task scf energy")
        assert task_count == 3  # dimer + 2 counterpoise

    def test_bsse_ghost_atoms(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_bsse

        write_nwchem_bsse(water_dimer_fragments, tmp_path / "t.nw")
        content = _read(tmp_path, "t.nw")

        # Ghost atoms are prefixed with "bq"
        assert "bqo" in content.lower() or "bqh" in content.lower()

    def test_bsse_real_and_ghost_partitioning(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.nwchem import write_nwchem_bsse

        write_nwchem_bsse(water_dimer_fragments, tmp_path / "t.nw")
        content = _read(tmp_path, "t.nw")

        # Each monomer section should have both real and ghost atoms
        # Find geometry blocks after the first dimer one
        geom_blocks = re.findall(
            r"geometry units angstroms noautoz\n(.+?)end",
            content, re.DOTALL,
        )

        # Should have 3 geometry blocks: dimer, mono1, mono2
        assert len(geom_blocks) == 3

        # Monomer blocks should have ghost atoms
        for block in geom_blocks[1:]:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            has_real = any(not l.startswith("bq") for l in lines)
            has_ghost = any(l.startswith("bq") for l in lines)
            assert has_real
            assert has_ghost


# ===========================================================================
# ORCA
# ===========================================================================

class TestORCAWriter:
    """Test ORCA writer."""

    def test_write_orca_fragment(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        output_file = tmp_path / "dimer.inp"
        write_orca_fragment(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        assert "!" in content
        assert "* xyz" in content

    def test_write_orca_multijob(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_multijob

        output_file = tmp_path / "dimer.inp"
        write_orca_multijob(water_dimer_fragments, output_file)

        content = output_file.read_text()
        assert "$new_job" in content


class TestORCAFormatCorrectness:
    """Thorough format-correctness tests for ORCA writers."""

    def test_keywords_line(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp", method="B3LYP", basis="def2-TZVP")
        content = _read(tmp_path, "t.inp")

        assert "! B3LYP def2-TZVP" in content

    def test_keywords_with_runtype(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp", runtype="Opt")
        content = _read(tmp_path, "t.inp")

        assert "Opt" in content

    def test_keywords_with_extra(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(
            water_dimer_fragments, tmp_path / "t.inp",
            extra_keywords=["TightSCF", "RIJCOSX"],
        )
        content = _read(tmp_path, "t.inp")

        assert "TightSCF" in content
        assert "RIJCOSX" in content

    def test_coord_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "* xyz 0 1" in content
        # Ends with lone "*"
        lines = content.strip().split("\n")
        assert lines[-1].strip() == "*"

    def test_memory_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp", memory=8000)
        content = _read(tmp_path, "t.inp")

        assert "%maxcore 8000" in content

    def test_parallel_block_present(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp", nprocs=4)
        content = _read(tmp_path, "t.inp")

        assert "%pal" in content
        assert "nprocs 4" in content

    def test_parallel_block_absent_single_proc(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_fragment

        write_orca_fragment(water_dimer_fragments, tmp_path / "t.inp", nprocs=1)
        content = _read(tmp_path, "t.inp")

        assert "%pal" not in content

    def test_multijob_separators(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_multijob

        write_orca_multijob(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        # 2 fragments + 1 full system = 3 jobs, 2 separators
        assert content.count("$new_job") == 2

    def test_multijob_full_system_at_end(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.orca import write_orca_multijob

        write_orca_multijob(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "Full system" in content
        # Full system job should be last
        last_new_job = content.rfind("$new_job")
        full_system = content.find("Full system")
        assert full_system > last_new_job


# ===========================================================================
# Molpro
# ===========================================================================

class TestMolproFormatCorrectness:
    """Format-correctness tests for Molpro writers."""

    def test_sapt_validation(self, tmp_path, water_fragment):
        from autofragment.io.writers.molpro import write_molpro_sapt

        with pytest.raises(ValueError, match="exactly 2"):
            write_molpro_sapt([water_fragment], tmp_path / "t.com")

    def test_sapt_three_frags_invalid(self, tmp_path, three_fragment_system):
        from autofragment.io.writers.molpro import write_molpro_sapt

        with pytest.raises(ValueError, match="exactly 2"):
            write_molpro_sapt(three_fragment_system, tmp_path / "t.com")

    def test_sapt_memory(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.molpro import write_molpro_sapt

        write_molpro_sapt(water_dimer_fragments, tmp_path / "t.com", memory=1000)
        content = _read(tmp_path, "t.com")

        assert "memory,1000,m" in content

    def test_sapt_geometry_comma_coords(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.molpro import write_molpro_sapt

        write_molpro_sapt(water_dimer_fragments, tmp_path / "t.com")
        content = _read(tmp_path, "t.com")

        assert "geometry={" in content
        assert "}" in content

        # Coordinates should be comma-separated
        geom_match = re.search(r"geometry=\{(.+?)\}", content, re.DOTALL)
        assert geom_match
        geom_block = geom_match.group(1)
        # Atom lines have commas between coordinates
        for line in geom_block.strip().split("\n"):
            if line.strip():
                assert "," in line

    def test_sapt_monomer_atom_comments(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.molpro import write_molpro_sapt

        write_molpro_sapt(water_dimer_fragments, tmp_path / "t.com")
        content = _read(tmp_path, "t.com")

        assert "! Monomer A atoms: 1,2,3" in content
        assert "! Monomer B atoms: 4,5,6" in content

    def test_sapt_hf_and_sapt_directives(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.molpro import write_molpro_sapt

        write_molpro_sapt(water_dimer_fragments, tmp_path / "t.com")
        content = _read(tmp_path, "t.com")

        assert "{hf}" in content
        assert "{df-sapt}" in content

    def test_lmp2_hf_and_lmp2_directives(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.molpro import write_molpro_lmp2

        write_molpro_lmp2(water_dimer_fragments, tmp_path / "t.com")
        content = _read(tmp_path, "t.com")

        assert "{hf}" in content
        assert "{df-lmp2}" in content


# ===========================================================================
# Turbomole
# ===========================================================================

class TestTurbomoleFormatCorrectness:
    """Format-correctness tests for Turbomole writer."""

    def test_coord_block(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.turbomole import write_turbomole_fragment

        write_turbomole_fragment(water_dimer_fragments, tmp_path / "coord")
        content = _read(tmp_path, "coord")

        assert "$coord" in content
        assert "$end" in content

    def test_bohr_conversion(self, tmp_path, water_fragment):
        from autofragment.io.writers.turbomole import write_turbomole_fragment

        write_turbomole_fragment([water_fragment], tmp_path / "coord")
        content = _read(tmp_path, "coord")

        # H at 0.958 Angstrom -> 0.958 * 1.8897259886 = 1.8103... Bohr
        expected_bohr = 0.958 * 1.8897259886
        # Should appear in the output with high precision
        assert f"{expected_bohr:.14f}" in content or f"{expected_bohr:.13f}" in content[:len(content)]

    def test_element_lowercase(self, tmp_path, water_fragment):
        from autofragment.io.writers.turbomole import write_turbomole_fragment

        write_turbomole_fragment([water_fragment], tmp_path / "coord")
        content = _read(tmp_path, "coord")

        # Elements should be lowercase
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("$"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                element = parts[3]
                assert element == element.lower(), f"Element not lowercase: {element}"

    def test_coord_precision(self, tmp_path, water_fragment):
        from autofragment.io.writers.turbomole import write_turbomole_fragment

        write_turbomole_fragment([water_fragment], tmp_path / "coord")
        content = _read(tmp_path, "coord")

        # Should have 18.14f precision format
        # Check that coordinates have many decimal places
        coord_pattern = re.compile(r"-?\d+\.\d{14}")
        assert coord_pattern.search(content)

    def test_control_hint_file(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.turbomole import write_turbomole_fragment

        write_turbomole_fragment(water_dimer_fragments, tmp_path / "coord", basis="def2-SVP", method="dft")
        hint_path = tmp_path / "control.hint"

        assert hint_path.exists()
        content = hint_path.read_text()

        assert "def2-SVP" in content
        assert "dft" in content
        assert "Fragments: 2" in content


# ===========================================================================
# CFOUR
# ===========================================================================

class TestCFOURFormatCorrectness:
    """Format-correctness tests for CFOUR writer."""

    def test_structure(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.cfour import write_cfour_fragment

        write_cfour_fragment(water_dimer_fragments, tmp_path / "ZMAT")
        content = _read(tmp_path, "ZMAT")
        lines = content.strip().split("\n")

        # First line is title
        assert "AutoFragment" in lines[0]

        # After atoms, blank line, then *CFOUR(...)
        assert "*CFOUR(" in content

    def test_keywords(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.cfour import write_cfour_fragment

        write_cfour_fragment(
            water_dimer_fragments, tmp_path / "ZMAT", method="CCSD(T)", basis="PVTZ"
        )
        content = _read(tmp_path, "ZMAT")

        assert "CALC_LEVEL=CCSD(T)" in content
        assert "BASIS=PVTZ" in content
        assert "CHARGE=0" in content
        assert "MULTIPLICITY=1" in content
        assert "COORDINATES=CARTESIAN" in content
        assert "UNITS=ANGSTROM" in content

    def test_blank_line_separator(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.cfour import write_cfour_fragment

        write_cfour_fragment(water_dimer_fragments, tmp_path / "ZMAT")
        content = _read(tmp_path, "ZMAT")

        # Blank line separates geometry from keywords
        assert "\n\n*CFOUR(" in content


# ===========================================================================
# XYZ
# ===========================================================================

class TestXYZWriter:
    """Test XYZ writer with fragment markers."""

    def test_write_xyz_fragments(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        output_file = tmp_path / "dimer.xyz"
        write_xyz_fragments(water_dimer_fragments, output_file)

        assert output_file.exists()
        content = output_file.read_text()

        lines = content.strip().split("\n")
        assert lines[0] == "6"
        assert "Fragments" in lines[1]


class TestXYZFormatCorrectness:
    """Thorough format-correctness tests for XYZ writers."""

    def test_atom_count(self, tmp_path, three_fragment_system):
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        write_xyz_fragments(three_fragment_system, tmp_path / "t.xyz")
        content = _read(tmp_path, "t.xyz")

        lines = content.strip().split("\n")
        assert lines[0].strip() == "9"

    def test_comment_with_summary(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        write_xyz_fragments(water_dimer_fragments, tmp_path / "t.xyz", include_summary=True)
        content = _read(tmp_path, "t.xyz")

        lines = content.strip().split("\n")
        assert "W1(3 atoms)" in lines[1]
        assert "W2(3 atoms)" in lines[1]

    def test_fragment_comments_appended(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        write_xyz_fragments(
            water_dimer_fragments, tmp_path / "t.xyz", include_fragment_comments=True
        )
        content = _read(tmp_path, "t.xyz")

        lines = content.strip().split("\n")
        # Atom lines (lines 2+) should have fragment ID comments
        assert "# W1" in lines[2]
        assert "# W2" in lines[5]

    def test_no_fragment_comments(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_fragments

        write_xyz_fragments(
            water_dimer_fragments, tmp_path / "t.xyz", include_fragment_comments=False
        )
        content = _read(tmp_path, "t.xyz")

        assert "# W1" not in content

    def test_extended_properties_line(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_extended

        write_xyz_extended(water_dimer_fragments, tmp_path / "t.xyz")
        content = _read(tmp_path, "t.xyz")

        lines = content.strip().split("\n")
        assert "Properties=species:S:1:pos:R:3:fragment:S:1" in lines[1]

    def test_extended_with_lattice(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_extended

        lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        write_xyz_extended(water_dimer_fragments, tmp_path / "t.xyz", lattice=lattice)
        content = _read(tmp_path, "t.xyz")

        assert 'Lattice="' in content

    def test_extended_with_properties(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.xyz_writer import write_xyz_extended

        props = {"charge": [0.0, 0.1, 0.1, 0.0, 0.1, 0.1]}
        write_xyz_extended(water_dimer_fragments, tmp_path / "t.xyz", properties=props)
        content = _read(tmp_path, "t.xyz")

        assert "charge:R:1" in content


# ===========================================================================
# PDB
# ===========================================================================

class TestPDBWriter:
    """Test PDB writer with fragment encoding."""

    def test_write_pdb_chains(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        output_file = tmp_path / "dimer.pdb"
        write_pdb_fragments(water_dimer_fragments, output_file, mode="chains")

        assert output_file.exists()
        content = output_file.read_text()

        assert "ATOM" in content
        assert " A " in content
        assert " B " in content
        assert "END" in content

    def test_write_pdb_models(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        output_file = tmp_path / "dimer.pdb"
        write_pdb_fragments(water_dimer_fragments, output_file, mode="models")

        content = output_file.read_text()
        assert "MODEL" in content
        assert "ENDMDL" in content


class TestPDBFormatCorrectness:
    """Thorough format-correctness tests for PDB writer."""

    def test_atom_record_width(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(water_dimer_fragments, tmp_path / "t.pdb", mode="chains")
        content = _read(tmp_path, "t.pdb")

        for line in content.split("\n"):
            if line.startswith("ATOM"):
                # PDB spec is 80 chars; writer omits altLoc so produces 79
                assert len(line) in (79, 80), f"ATOM record bad width: {len(line)} '{line}'"

    def test_chains_mode_chain_ids(self, tmp_path, three_fragment_system):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(three_fragment_system, tmp_path / "t.pdb", mode="chains")
        content = _read(tmp_path, "t.pdb")

        # Should have chains A, B, C
        # Extract chain IDs from ATOM lines by pattern
        atom_lines = [l for l in content.split("\n") if l.startswith("ATOM")]
        chain_ids = set()
        for line in atom_lines:
            # Chain ID follows the residue name in the format string
            # Match the pattern: "ATOM" ... "FRG X" where X is chain
            m = re.search(r"[A-Z]{3}\s+([A-Z])", line[15:25])
            if m:
                chain_ids.add(m.group(1))

        assert "A" in chain_ids
        assert "B" in chain_ids
        assert "C" in chain_ids

    def test_chains_mode_ter_records(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(water_dimer_fragments, tmp_path / "t.pdb", mode="chains")
        content = _read(tmp_path, "t.pdb")

        assert content.count("TER") >= 2
        assert "END" in content

    def test_models_mode_structure(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(water_dimer_fragments, tmp_path / "t.pdb", mode="models")
        content = _read(tmp_path, "t.pdb")

        model_lines = [l for l in content.split("\n") if l.startswith("MODEL ") or l.startswith("MODEL\t")]
        assert len(model_lines) == 2
        assert content.count("ENDMDL") == 2
        assert content.strip().endswith("END")

    def test_residues_mode(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(water_dimer_fragments, tmp_path / "t.pdb", mode="residues")
        content = _read(tmp_path, "t.pdb")

        atom_lines = [l for l in content.split("\n") if l.startswith("ATOM")]

        # All atoms should mention chain A and have residue W1 or W2
        for line in atom_lines:
            assert " A " in line

        # Different residue names: W1, W2
        res_names = set()
        for line in atom_lines:
            # Residue name is right-justified in 3 chars before chain ID
            m = re.search(r"(\w+)\s+A\s+\d+", line)
            if m:
                res_names.add(m.group(1))

        assert len(res_names) == 2

    def test_coordinate_precision(self, tmp_path, water_fragment):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments([water_fragment], tmp_path / "t.pdb", mode="chains")
        content = _read(tmp_path, "t.pdb")

        # PDB standard: 8.3f
        for line in content.split("\n"):
            if line.startswith("ATOM"):
                x_str = line[30:38].strip()
                # Should have 3 decimal places
                assert "." in x_str
                decimals = x_str.split(".")[1]
                assert len(decimals) == 3

    def test_invalid_mode_raises(self, tmp_path, water_fragment):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        with pytest.raises(ValueError, match="Unknown mode"):
            write_pdb_fragments([water_fragment], tmp_path / "t.pdb", mode="invalid")


# ===========================================================================
# QCSchema
# ===========================================================================

class TestQCSchemaWriter:
    """Test QCSchema writer."""

    def test_write_qcschema(self, tmp_path, water_fragment):
        from autofragment.io.writers.qcschema_writer import write_qcschema

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
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        atoms = [Atom(symbol="H", coords=np.array([0.0, 0.0, 0.0]))]
        system = ChemicalSystem(atoms=atoms)

        result = system_to_qcschema(system)

        assert result["geometry"] == [0.0, 0.0, 0.0]


class TestQCSchemaFormatCorrectness:
    """Thorough format-correctness tests for QCSchema writers."""

    def test_angstrom_to_bohr_conversion(self):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        atoms = [Atom(symbol="H", coords=np.array([1.0, 0.0, 0.0]))]
        system = ChemicalSystem(atoms=atoms)

        result = system_to_qcschema(system)

        expected_bohr = 1.0 * 1.8897259886
        assert abs(result["geometry"][0] - expected_bohr) < 1e-6

    def test_schema_fields(self):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        atoms = [Atom(symbol="O", coords=np.array([0.0, 0.0, 0.0]))]
        system = ChemicalSystem(atoms=atoms)

        result = system_to_qcschema(system)

        assert result["schema_name"] == "qcschema_molecule"
        assert result["schema_version"] == 2
        assert result["symbols"] == ["O"]
        assert "geometry" in result

    def test_connectivity_from_bonds(self, water_dimer_system):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        water_dimer_system.bonds = [
            {"atom1": 0, "atom2": 1, "order": 1.0},
            {"atom1": 0, "atom2": 2, "order": 1.0},
        ]

        result = system_to_qcschema(water_dimer_system)

        assert "connectivity" in result
        assert len(result["connectivity"]) == 2
        assert result["connectivity"][0] == [0, 1, 1.0]

    def test_fragment_indices(self, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        result = system_to_qcschema(water_dimer_system, fragments=water_dimer_fragments)

        assert "fragments" in result
        assert len(result["fragments"]) == 2

    def test_provenance(self):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        atoms = [Atom(symbol="H", coords=np.array([0.0, 0.0, 0.0]))]
        system = ChemicalSystem(atoms=atoms)

        result = system_to_qcschema(system)

        assert result["provenance"]["creator"] == "autofragment"

    def test_input_spec(self, tmp_path, water_dimer_system):
        from autofragment.io.writers.qcschema_writer import write_qcschema_input

        write_qcschema_input(
            water_dimer_system, tmp_path / "t.json",
            driver="gradient", method="b3lyp", basis="def2-svp"
        )

        with open(tmp_path / "t.json") as f:
            data = json.load(f)

        assert data["schema_name"] == "qcschema_input"
        assert data["schema_version"] == 1
        assert data["driver"] == "gradient"
        assert data["model"]["method"] == "b3lyp"
        assert data["model"]["basis"] == "def2-svp"

    def test_input_spec_has_molecule(self, tmp_path, water_dimer_system):
        from autofragment.io.writers.qcschema_writer import write_qcschema_input

        write_qcschema_input(water_dimer_system, tmp_path / "t.json")

        with open(tmp_path / "t.json") as f:
            data = json.load(f)

        assert "molecule" in data
        assert data["molecule"]["schema_name"] == "qcschema_molecule"


# ===========================================================================
# Format Registry
# ===========================================================================

class TestFormatRegistry:
    """Test format registry system."""

    def test_registry_supported_formats(self):
        from autofragment.io.writers.registry import FormatRegistry

        formats = FormatRegistry.supported_formats()

        assert "read" in formats
        assert "write" in formats

    def test_registry_custom_format(self):
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

        FormatRegistry.unregister_reader("custom")


# ===========================================================================
# Phase 3: GAMESS $FMOBND interfragment bonds
# ===========================================================================

class TestGAMESSInterfragmentBonds:
    """Tests for GAMESS $FMOBND interfragment bond support."""

    def test_fmobnd_present(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_gamess_fmo(
            [frag1, frag2], tmp_path / "t.inp",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.inp")

        assert "$FMOBND" in content
        assert "$END" in content

    def test_fmobnd_indices_1based(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_gamess_fmo(
            [frag1, frag2], tmp_path / "t.inp",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.inp")

        # bond: RES1 atom 2 (0-based) -> 3 (1-based), frag "RES1" -> 1
        #        RES2 atom 0 (0-based) -> 1 (1-based), frag "RES2" -> 2
        # FMO convention: first atom gets negative sign
        assert "  -3 1 1 2" in content

    def test_fmobnd_multiple_bonds(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        frag1, frag2, bonds = bonded_dipeptide_fragments
        second_bond = {
            "fragment1_id": "RES1",
            "atom1_index": 0,
            "fragment2_id": "RES2",
            "atom2_index": 4,
            "bond_order": 1.0,
            "metadata": {},
        }
        write_gamess_fmo(
            [frag1, frag2], tmp_path / "t.inp",
            interfragment_bonds=bonds + [second_bond],
        )
        content = _read(tmp_path, "t.inp")

        # Extract FMOBND block
        fmobnd_match = re.search(r"\$FMOBND\n(.+?)\s*\$END", content, re.DOTALL)
        assert fmobnd_match
        bond_lines = [l.strip() for l in fmobnd_match.group(1).strip().split("\n") if l.strip()]
        assert len(bond_lines) == 2

    def test_no_fmobnd_without_bonds(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(water_dimer_fragments, tmp_path / "t.inp")
        content = _read(tmp_path, "t.inp")

        assert "$FMOBND" not in content

    def test_fmobnd_empty_list(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        write_gamess_fmo(
            water_dimer_fragments, tmp_path / "t.inp",
            interfragment_bonds=[],
        )
        content = _read(tmp_path, "t.inp")

        assert "$FMOBND" not in content

    def test_fmobnd_unknown_fragment_raises(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.gamess import write_gamess_fmo

        bad_bonds = [{
            "fragment1_id": "NONEXISTENT",
            "atom1_index": 0,
            "fragment2_id": "W1",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {},
        }]

        with pytest.raises(ValueError, match="unknown fragment"):
            write_gamess_fmo(
                water_dimer_fragments, tmp_path / "t.inp",
                interfragment_bonds=bad_bonds,
            )

    def test_efmo_passthrough(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.gamess import write_gamess_efmo

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_gamess_efmo(
            [frag1, frag2], tmp_path / "t.inp",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.inp")

        assert "$FMOBND" in content
        assert "MODGRD=1" in content


# ===========================================================================
# Phase 4: PDB CONECT records
# ===========================================================================

class TestPDBInterfragmentBonds:
    """Tests for PDB CONECT record support."""

    def test_conect_in_chains_mode(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_pdb_fragments(
            [frag1, frag2], tmp_path / "t.pdb", mode="chains",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.pdb")

        assert "CONECT" in content

        # Find CONECT lines
        conect_lines = [l for l in content.split("\n") if l.startswith("CONECT")]
        assert len(conect_lines) >= 1

        # Bond is frag1 atom 2 -> serial 3 (1-based, frag1 starts at serial 1)
        # Bond is frag2 atom 0 -> serial 9 (frag2 starts after frag1's 7 atoms + TER = serial 9)
        # In chains mode with TER records incrementing serial:
        # frag1: serials 1-7, TER at 8; frag2: serials 9-15, TER at 16
        # atom1 = serial 3, atom2 = serial 9

    def test_conect_in_residues_mode(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_pdb_fragments(
            [frag1, frag2], tmp_path / "t.pdb", mode="residues",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.pdb")

        assert "CONECT" in content

    def test_no_conect_without_bonds(self, tmp_path, water_dimer_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        write_pdb_fragments(water_dimer_fragments, tmp_path / "t.pdb", mode="chains")
        content = _read(tmp_path, "t.pdb")

        assert "CONECT" not in content

    def test_models_mode_no_conect(self, tmp_path, bonded_dipeptide_fragments):
        from autofragment.io.writers.pdb_writer import write_pdb_fragments

        frag1, frag2, bonds = bonded_dipeptide_fragments
        write_pdb_fragments(
            [frag1, frag2], tmp_path / "t.pdb", mode="models",
            interfragment_bonds=bonds,
        )
        content = _read(tmp_path, "t.pdb")

        # Models mode silently skips CONECT
        assert "CONECT" not in content


# ===========================================================================
# Phase 5: QCSchema connectivity
# ===========================================================================

class TestQCSchemaInterfragmentBonds:
    """Tests for QCSchema interfragment bond connectivity support."""

    def test_connectivity_includes_interfragment(self, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        ifbonds = [{
            "fragment1_id": "W1",
            "atom1_index": 0,
            "fragment2_id": "W2",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {},
        }]

        result = system_to_qcschema(
            water_dimer_system, fragments=water_dimer_fragments,
            interfragment_bonds=ifbonds,
        )

        assert "connectivity" in result
        # Global indices: W1 atom 0 -> global 0, W2 atom 0 -> global 3
        found = any(c[0] == 0 and c[1] == 3 and c[2] == 1.0 for c in result["connectivity"])
        assert found, f"Expected [0, 3, 1.0] in connectivity: {result['connectivity']}"

    def test_no_duplicate_connectivity(self, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        # Add the bond to system.bonds AND interfragment_bonds
        water_dimer_system.bonds = [{"atom1": 0, "atom2": 3, "order": 1.0}]

        ifbonds = [{
            "fragment1_id": "W1",
            "atom1_index": 0,
            "fragment2_id": "W2",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {},
        }]

        result = system_to_qcschema(
            water_dimer_system, fragments=water_dimer_fragments,
            interfragment_bonds=ifbonds,
        )

        # Should not have duplicate [0, 3, 1.0]
        matching = [c for c in result["connectivity"] if c[0] == 0 and c[1] == 3]
        assert len(matching) == 1

    def test_no_connectivity_without_bonds(self, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import system_to_qcschema

        result = system_to_qcschema(water_dimer_system, fragments=water_dimer_fragments)

        # No bonds in system and no interfragment_bonds -> no connectivity key
        assert "connectivity" not in result

    def test_write_qcschema_passthrough(self, tmp_path, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import write_qcschema

        ifbonds = [{
            "fragment1_id": "W1",
            "atom1_index": 0,
            "fragment2_id": "W2",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {},
        }]

        write_qcschema(
            water_dimer_system, tmp_path / "t.json",
            fragments=water_dimer_fragments,
            interfragment_bonds=ifbonds,
        )

        with open(tmp_path / "t.json") as f:
            data = json.load(f)

        assert "connectivity" in data
        found = any(c[0] == 0 and c[1] == 3 and c[2] == 1.0 for c in data["connectivity"])
        assert found

    def test_write_qcschema_input_passthrough(self, tmp_path, water_dimer_system, water_dimer_fragments):
        from autofragment.io.writers.qcschema_writer import write_qcschema_input

        ifbonds = [{
            "fragment1_id": "W1",
            "atom1_index": 0,
            "fragment2_id": "W2",
            "atom2_index": 0,
            "bond_order": 1.0,
            "metadata": {},
        }]

        write_qcschema_input(
            water_dimer_system, tmp_path / "t.json",
            fragments=water_dimer_fragments,
            interfragment_bonds=ifbonds,
        )

        with open(tmp_path / "t.json") as f:
            data = json.load(f)

        mol = data["molecule"]
        assert "connectivity" in mol
