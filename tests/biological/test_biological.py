# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for biological system components (Phase 5).

Tests for:
- Amino acid charge data
- Nucleotide data
- pH calculations (Henderson-Hasselbalch)
- BioPartitioner enhancements
- NucleicPartitioner
"""


import pytest


class TestAminoAcidData:
    """Tests for amino acid data module."""

    def test_amino_acid_count(self):
        """All 20 standard amino acids plus variants are defined."""
        from autofragment.data.amino_acids import AMINO_ACIDS

        # 20 standard + HID, HIE, HIP (histidine variants) + MSE, SEC (selenium)
        assert len(AMINO_ACIDS) >= 25

    def test_standard_amino_acids_present(self):
        """All 20 standard amino acids are present."""
        from autofragment.data.amino_acids import AMINO_ACIDS

        standard = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        ]
        for aa in standard:
            assert aa in AMINO_ACIDS, f"Missing amino acid: {aa}"

    def test_selenium_amino_acids(self):
        """Selenomethionine and selenocysteine are defined."""
        from autofragment.data.amino_acids import AMINO_ACIDS

        assert "MSE" in AMINO_ACIDS
        assert "SEC" in AMINO_ACIDS
        assert AMINO_ACIDS["MSE"].name == "Selenomethionine"
        assert AMINO_ACIDS["SEC"].name == "Selenocysteine"

    def test_amino_acid_properties(self):
        """Amino acid properties are correctly set."""
        from autofragment.data.amino_acids import AMINO_ACIDS

        # Alanine should be non-polar, non-charged, non-aromatic
        ala = AMINO_ACIDS["ALA"]
        assert ala.is_polar is False
        assert ala.is_charged is False
        assert ala.is_aromatic is False

        # Aspartic acid should be polar, charged
        asp = AMINO_ACIDS["ASP"]
        assert asp.is_polar is True
        assert asp.is_charged is True
        assert asp.pka_sidechain == pytest.approx(3.65, abs=0.1)

        # Phenylalanine should be aromatic
        phe = AMINO_ACIDS["PHE"]
        assert phe.is_aromatic is True

    def test_molecular_weights(self):
        """Molecular weights are reasonable."""
        from autofragment.data.amino_acids import AMINO_ACIDS

        # Glycine is smallest
        assert AMINO_ACIDS["GLY"].mw == pytest.approx(75.07, abs=1)
        # Tryptophan is largest standard
        assert AMINO_ACIDS["TRP"].mw == pytest.approx(204.23, abs=1)

    def test_get_amino_acid_function(self):
        """get_amino_acid function works correctly."""
        from autofragment.data.amino_acids import get_amino_acid

        aa = get_amino_acid("ALA")
        assert aa is not None
        assert aa.name == "Alanine"

        # Case insensitive
        aa = get_amino_acid("ala")
        assert aa is not None

        # Unknown returns None
        assert get_amino_acid("XYZ") is None


class TestChargeCalculations:
    """Tests for pH-dependent charge calculations."""

    def test_charge_at_ph74_acidic(self):
        """Acidic residues have correct charge at pH 7.4."""
        from autofragment.data.amino_acids import get_charge_at_ph74

        assert get_charge_at_ph74("ASP") == -1.0
        assert get_charge_at_ph74("GLU") == -1.0

    def test_charge_at_ph74_basic(self):
        """Basic residues have correct charge at pH 7.4."""
        from autofragment.data.amino_acids import get_charge_at_ph74

        assert get_charge_at_ph74("LYS") == +1.0
        assert get_charge_at_ph74("ARG") == +1.0

    def test_charge_at_ph74_neutral(self):
        """Neutral residues have zero charge at pH 7.4."""
        from autofragment.data.amino_acids import get_charge_at_ph74

        for aa in ["ALA", "GLY", "VAL", "LEU", "ILE", "PRO", "PHE"]:
            assert get_charge_at_ph74(aa) == 0.0

    def test_histidine_partial_charge(self):
        """Histidine has partial charge at pH 7.4."""
        from autofragment.data.amino_acids import get_charge_at_ph74

        # HIS pKa is ~6.0, so at pH 7.4 it's ~10% protonated
        his_charge = get_charge_at_ph74("HIS")
        assert 0.0 < his_charge < 0.2

    def test_henderson_hasselbalch_acidic(self):
        """Henderson-Hasselbalch equation for acidic groups."""
        from autofragment.chemistry import henderson_hasselbalch_acidic

        # At pKa, 50% deprotonated
        assert henderson_hasselbalch_acidic(3.65, 3.65) == pytest.approx(-0.5, abs=0.01)

        # Far above pKa, fully deprotonated
        assert henderson_hasselbalch_acidic(7.0, 3.65) == pytest.approx(-1.0, abs=0.01)

        # Far below pKa, fully protonated
        assert henderson_hasselbalch_acidic(1.0, 3.65) == pytest.approx(0.0, abs=0.01)

    def test_henderson_hasselbalch_basic(self):
        """Henderson-Hasselbalch equation for basic groups."""
        from autofragment.chemistry import henderson_hasselbalch_basic

        # At pKa, 50% protonated
        assert henderson_hasselbalch_basic(10.53, 10.53) == pytest.approx(0.5, abs=0.01)

        # Far below pKa, fully protonated
        assert henderson_hasselbalch_basic(7.0, 10.53) == pytest.approx(1.0, abs=0.01)

        # Far above pKa, fully deprotonated
        assert henderson_hasselbalch_basic(14.0, 10.53) == pytest.approx(0.0, abs=0.01)

    def test_residue_charge_at_ph(self):
        """Test pH-dependent residue charge calculation."""
        from autofragment.data.amino_acids import get_residue_charge_at_ph

        # ASP at pH 2 (below pKa 3.65) - mostly protonated
        charge = get_residue_charge_at_ph("ASP", 2.0)
        assert -0.2 < charge < 0.0

        # ASP at pH 7 (far above pKa) - fully deprotonated
        charge = get_residue_charge_at_ph("ASP", 7.0)
        assert charge < -0.99

        # LYS at pH 12 (above pKa 10.53) - mostly deprotonated
        charge = get_residue_charge_at_ph("LYS", 12.0)
        assert charge < 0.1

    def test_protein_charge_calculation(self):
        """Test protein charge calculation."""
        from autofragment.data.amino_acids import calculate_protein_charge

        # Simple peptide: Ala-Asp-Lys (no terminals)
        charge = calculate_protein_charge(
            ["ALA", "ASP", "LYS"],
            ph=7.4,
            n_terminus=False,
            c_terminus=False
        )
        # ASP: -1, LYS: +1, ALA: 0 = net 0
        assert charge == pytest.approx(0.0, abs=0.1)


class TestPKAValues:
    """Tests for pKa lookup tables."""

    def test_pka_lookup(self):
        """Test pKa value lookup."""
        from autofragment.data.pka_values import get_pka

        assert get_pka("ASP") == pytest.approx(3.65)
        assert get_pka("ASP_SIDECHAIN") == pytest.approx(3.65)
        assert get_pka("N_TERMINUS") == pytest.approx(7.7)
        assert get_pka("PHOSPHATE_PKA1") == pytest.approx(1.0)

    def test_terminal_pka(self):
        """Test terminal pKa with residue adjustments."""
        from autofragment.data.pka_values import get_terminal_pka

        base_n = get_terminal_pka("N_TERMINUS")
        gly_n = get_terminal_pka("N_TERMINUS", "GLY")

        # Glycine should have adjustment
        assert gly_n != base_n


class TestPTMData:
    """Tests for post-translational modification data."""

    def test_phosphorylation_charge(self):
        """Phosphorylated residues have -2 charge change."""
        from autofragment.data.amino_acids import PTM_DATABASE, get_ptm_charge_adjustment

        assert "SEP" in PTM_DATABASE  # Phosphoserine
        assert "TPO" in PTM_DATABASE  # Phosphothreonine
        assert "PTR" in PTM_DATABASE  # Phosphotyrosine

        assert get_ptm_charge_adjustment("SEP") == -2.0
        assert get_ptm_charge_adjustment("TPO") == -2.0
        assert get_ptm_charge_adjustment("PTR") == -2.0

    def test_acetylation_charge(self):
        """Acetyllysine has -1 charge change."""
        from autofragment.data.amino_acids import get_ptm_charge_adjustment

        # Acetylation neutralizes the lysine positive charge
        assert get_ptm_charge_adjustment("ALY") == -1.0


class TestNucleotideData:
    """Tests for nucleotide data module."""

    def test_dna_nucleotides(self):
        """All 4 DNA nucleotides are defined."""
        from autofragment.data.nucleotides import DNA_NUCLEOTIDES

        assert "DA" in DNA_NUCLEOTIDES
        assert "DT" in DNA_NUCLEOTIDES
        assert "DG" in DNA_NUCLEOTIDES
        assert "DC" in DNA_NUCLEOTIDES

    def test_rna_nucleotides(self):
        """All 4 RNA nucleotides are defined."""
        from autofragment.data.nucleotides import RNA_NUCLEOTIDES

        assert "A" in RNA_NUCLEOTIDES
        assert "U" in RNA_NUCLEOTIDES
        assert "G" in RNA_NUCLEOTIDES
        assert "C" in RNA_NUCLEOTIDES

    def test_nucleotide_charge(self):
        """Nucleotides have -1 backbone charge."""
        from autofragment.data.nucleotides import DNA_NUCLEOTIDES

        for code, nuc in DNA_NUCLEOTIDES.items():
            assert nuc.charge_backbone == -1.0
            assert nuc.charge_base == 0.0
            assert nuc.charge_sugar == 0.0
            assert nuc.total_charge == -1.0

    def test_nucleic_acid_charge_calculation(self):
        """Test nucleic acid chain charge calculation."""
        from autofragment.data.nucleotides import get_nucleic_acid_charge

        # 4 nucleotides = 3 internal phosphates = -3 charge
        charge = get_nucleic_acid_charge(["DA", "DT", "DG", "DC"])
        assert charge == -3.0

        # With 5' phosphate
        charge = get_nucleic_acid_charge(
            ["DA", "DT", "DG", "DC"],
            include_5prime_phosphate=True
        )
        assert charge == -4.0

    def test_watson_crick_pairing(self):
        """Test Watson-Crick base pairing detection."""
        from autofragment.data.nucleotides import can_base_pair

        # DNA pairs
        assert can_base_pair("A", "T") == "Watson-Crick"
        assert can_base_pair("G", "C") == "Watson-Crick"

        # RNA pairs
        assert can_base_pair("A", "U") == "Watson-Crick"

        # Non-pairs
        assert can_base_pair("A", "G") is None
        assert can_base_pair("A", "A") is None

        # Wobble pair
        assert can_base_pair("G", "U") == "Wobble"


class TestNucleicPartitioner:
    """Tests for NucleicPartitioner class."""

    def test_partitioner_creation(self):
        """NucleicPartitioner can be instantiated."""
        from autofragment.partitioners.nucleic import NucleicPartitioner

        partitioner = NucleicPartitioner()
        assert partitioner.nucleotides_per_fragment == 3
        assert partitioner.preserve_base_pairs is True
        assert partitioner.partition_mode == "backbone"

    def test_partitioner_custom_parameters(self):
        """NucleicPartitioner accepts custom parameters."""
        from autofragment.partitioners.nucleic import NucleicPartitioner

        partitioner = NucleicPartitioner(
            nucleotides_per_fragment=5,
            preserve_base_pairs=False,
            partition_mode="base"
        )
        assert partitioner.nucleotides_per_fragment == 5
        assert partitioner.preserve_base_pairs is False
        assert partitioner.partition_mode == "base"

    def test_partitioner_validation(self):
        """NucleicPartitioner validates parameters."""
        from autofragment.partitioners.nucleic import NucleicPartitioner

        with pytest.raises(ValueError):
            NucleicPartitioner(nucleotides_per_fragment=0)

        with pytest.raises(ValueError):
            NucleicPartitioner(partition_mode="invalid")

    def test_partition_empty(self):
        """Partitioning empty residue list returns empty FragmentTree."""
        from autofragment.partitioners.nucleic import NucleicPartitioner

        partitioner = NucleicPartitioner()
        result = partitioner.partition_residues([])

        assert len(result.fragments) == 0


class TestIsoelectricPoint:
    """Tests for isoelectric point calculation."""

    def test_simple_pi_calculation(self):
        """Calculate pI for simple peptides."""
        from autofragment.chemistry.ph import calculate_isoelectric_point

        # Glycine-only peptide: pI near 6 (average of terminal pKas ~5.5)
        pi = calculate_isoelectric_point(["GLY"])
        assert 5.0 < pi < 6.5

    def test_acidic_protein_pi(self):
        """Acidic proteins have low pI."""
        from autofragment.chemistry.ph import calculate_isoelectric_point

        # Many acidic residues
        pi = calculate_isoelectric_point(["GLY", "ASP", "ASP", "GLU", "GLU"])
        assert pi < 4.5

    def test_basic_protein_pi(self):
        """Basic proteins have high pI."""
        from autofragment.chemistry.ph import calculate_isoelectric_point

        # Many basic residues
        pi = calculate_isoelectric_point(["GLY", "LYS", "LYS", "ARG", "ARG"])
        assert pi > 10


class TestPhValidation:
    """Tests for pH validation."""

    def test_valid_ph_range(self):
        """Valid pH values don't raise errors."""
        from autofragment.chemistry.ph import validate_ph

        validate_ph(0)
        validate_ph(7)
        validate_ph(14)

    def test_invalid_ph_raises(self):
        """Invalid pH values raise ValueError."""
        from autofragment.chemistry.ph import validate_ph

        with pytest.raises(ValueError):
            validate_ph(-1)

        with pytest.raises(ValueError):
            validate_ph(15)


class TestBioPartitionerPH:
    """Tests for BioPartitioner pH support."""

    def test_bio_partitioner_accepts_ph(self):
        """BioPartitioner accepts ph parameter."""
        from autofragment.partitioners.bio import BioPartitioner

        p = BioPartitioner(ph=4.0)
        assert p.ph == 4.0

    def test_bio_partitioner_default_ph(self):
        """BioPartitioner defaults to pH 7.4."""
        from autofragment.partitioners.bio import BioPartitioner

        p = BioPartitioner()
        assert p.ph == 7.4

    def test_bio_partitioner_invalid_ph(self):
        """BioPartitioner rejects invalid pH."""
        from autofragment.partitioners.bio import BioPartitioner

        with pytest.raises(ValueError):
            BioPartitioner(ph=-1)

        with pytest.raises(ValueError):
            BioPartitioner(ph=15)

    def test_sidechain_charge_ph74(self):
        """At pH 7.4, standard charges are as expected."""
        from autofragment.chemistry.ph import get_sidechain_charge

        assert round(get_sidechain_charge("ASP", 7.4)) == -1
        assert round(get_sidechain_charge("GLU", 7.4)) == -1
        assert round(get_sidechain_charge("LYS", 7.4)) == 1
        assert round(get_sidechain_charge("ARG", 7.4)) == 1
        assert round(get_sidechain_charge("HIS", 7.4)) == 0

    def test_sidechain_charge_low_ph(self):
        """At pH 4.0, HIS becomes +1 (pKa ~6.0, mostly protonated)."""
        from autofragment.chemistry.ph import get_sidechain_charge

        his_charge = get_sidechain_charge("HIS", 4.0)
        assert round(his_charge) == 1
