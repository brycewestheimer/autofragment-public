# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Amino acid charge and property data.

This module provides comprehensive data for all 20 standard amino acids,
plus selenium-containing amino acids and post-translational modifications.

Data includes:
- Molecular properties (formula, weight)
- pKa values for ionizable sidechains
- Charges at neutral, physiological (pH 7.4), acidic (pH 4), and basic (pH 10) conditions
- Physical properties (polarity, charge, aromaticity)

Reference values from:
- Stryer, Biochemistry
- CRC Handbook of Chemistry and Physics
- NIST Chemistry WebBook
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AminoAcidData:
    """Data for a single amino acid.

    Attributes
    ----------
    three_letter : str
        Three-letter code (e.g., "ALA")
    one_letter : str
        One-letter code (e.g., "A")
    name : str
        Full name (e.g., "Alanine")
    formula : str
        Molecular formula
    mw : float
        Molecular weight in Da
    charge_neutral : float
        Charge in neutral (protonated) form
    charge_ph7 : float
        Charge at physiological pH 7.4
    pka_sidechain : float or None
        pKa of ionizable side chain, if any
    is_polar : bool
        Whether the residue is polar
    is_charged : bool
        Whether the residue can be charged at physiological pH
    is_aromatic : bool
        Whether the residue has aromatic side chain
    """

    three_letter: str
    one_letter: str
    name: str
    formula: str
    mw: float
    charge_neutral: float
    charge_ph7: float
    pka_sidechain: Optional[float]
    is_polar: bool
    is_charged: bool
    is_aromatic: bool


# Standard 20 amino acids with complete data
AMINO_ACIDS: Dict[str, AminoAcidData] = {
    "ALA": AminoAcidData(
        three_letter="ALA",
        one_letter="A",
        name="Alanine",
        formula="C3H7NO2",
        mw=89.09,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "ARG": AminoAcidData(
        three_letter="ARG",
        one_letter="R",
        name="Arginine",
        formula="C6H14N4O2",
        mw=174.20,
        charge_neutral=0.0,
        charge_ph7=+1.0,
        pka_sidechain=12.48,
        is_polar=True,
        is_charged=True,
        is_aromatic=False,
    ),
    "ASN": AminoAcidData(
        three_letter="ASN",
        one_letter="N",
        name="Asparagine",
        formula="C4H8N2O3",
        mw=132.12,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
    "ASP": AminoAcidData(
        three_letter="ASP",
        one_letter="D",
        name="Aspartic acid",
        formula="C4H7NO4",
        mw=133.10,
        charge_neutral=0.0,
        charge_ph7=-1.0,
        pka_sidechain=3.65,
        is_polar=True,
        is_charged=True,
        is_aromatic=False,
    ),
    "CYS": AminoAcidData(
        three_letter="CYS",
        one_letter="C",
        name="Cysteine",
        formula="C3H7NO2S",
        mw=121.16,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=8.18,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
    "GLN": AminoAcidData(
        three_letter="GLN",
        one_letter="Q",
        name="Glutamine",
        formula="C5H10N2O3",
        mw=146.15,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
    "GLU": AminoAcidData(
        three_letter="GLU",
        one_letter="E",
        name="Glutamic acid",
        formula="C5H9NO4",
        mw=147.13,
        charge_neutral=0.0,
        charge_ph7=-1.0,
        pka_sidechain=4.25,
        is_polar=True,
        is_charged=True,
        is_aromatic=False,
    ),
    "GLY": AminoAcidData(
        three_letter="GLY",
        one_letter="G",
        name="Glycine",
        formula="C2H5NO2",
        mw=75.07,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "HIS": AminoAcidData(
        three_letter="HIS",
        one_letter="H",
        name="Histidine",
        formula="C6H9N3O2",
        mw=155.16,
        charge_neutral=0.0,
        charge_ph7=+0.1,
        pka_sidechain=6.00,
        is_polar=True,
        is_charged=False,
        is_aromatic=True,
    ),
    "ILE": AminoAcidData(
        three_letter="ILE",
        one_letter="I",
        name="Isoleucine",
        formula="C6H13NO2",
        mw=131.17,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "LEU": AminoAcidData(
        three_letter="LEU",
        one_letter="L",
        name="Leucine",
        formula="C6H13NO2",
        mw=131.17,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "LYS": AminoAcidData(
        three_letter="LYS",
        one_letter="K",
        name="Lysine",
        formula="C6H14N2O2",
        mw=146.19,
        charge_neutral=0.0,
        charge_ph7=+1.0,
        pka_sidechain=10.53,
        is_polar=True,
        is_charged=True,
        is_aromatic=False,
    ),
    "MET": AminoAcidData(
        three_letter="MET",
        one_letter="M",
        name="Methionine",
        formula="C5H11NO2S",
        mw=149.21,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "PHE": AminoAcidData(
        three_letter="PHE",
        one_letter="F",
        name="Phenylalanine",
        formula="C9H11NO2",
        mw=165.19,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=True,
    ),
    "PRO": AminoAcidData(
        three_letter="PRO",
        one_letter="P",
        name="Proline",
        formula="C5H9NO2",
        mw=115.13,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "SER": AminoAcidData(
        three_letter="SER",
        one_letter="S",
        name="Serine",
        formula="C3H7NO3",
        mw=105.09,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
    "THR": AminoAcidData(
        three_letter="THR",
        one_letter="T",
        name="Threonine",
        formula="C4H9NO3",
        mw=119.12,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
    "TRP": AminoAcidData(
        three_letter="TRP",
        one_letter="W",
        name="Tryptophan",
        formula="C11H12N2O2",
        mw=204.23,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=True,
    ),
    "TYR": AminoAcidData(
        three_letter="TYR",
        one_letter="Y",
        name="Tyrosine",
        formula="C9H11NO3",
        mw=181.19,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=10.07,
        is_polar=True,
        is_charged=False,
        is_aromatic=True,
    ),
    "VAL": AminoAcidData(
        three_letter="VAL",
        one_letter="V",
        name="Valine",
        formula="C5H11NO2",
        mw=117.15,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
}

# Histidine protonation variants (common in PDB files)
HISTIDINE_VARIANTS: Dict[str, AminoAcidData] = {
    "HID": AminoAcidData(
        three_letter="HID",
        one_letter="H",
        name="Histidine (delta-protonated)",
        formula="C6H9N3O2",
        mw=155.16,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=6.00,
        is_polar=True,
        is_charged=False,
        is_aromatic=True,
    ),
    "HIE": AminoAcidData(
        three_letter="HIE",
        one_letter="H",
        name="Histidine (epsilon-protonated)",
        formula="C6H9N3O2",
        mw=155.16,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=6.00,
        is_polar=True,
        is_charged=False,
        is_aromatic=True,
    ),
    "HIP": AminoAcidData(
        three_letter="HIP",
        one_letter="H",
        name="Histidine (doubly-protonated)",
        formula="C6H10N3O2",
        mw=156.16,
        charge_neutral=+1.0,
        charge_ph7=+1.0,
        pka_sidechain=6.00,
        is_polar=True,
        is_charged=True,
        is_aromatic=True,
    ),
}

# Selenium-containing amino acids (common in crystallography)
SELENIUM_AMINO_ACIDS: Dict[str, AminoAcidData] = {
    "MSE": AminoAcidData(
        three_letter="MSE",
        one_letter="M",
        name="Selenomethionine",
        formula="C5H11NO2Se",
        mw=196.11,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=None,
        is_polar=False,
        is_charged=False,
        is_aromatic=False,
    ),
    "SEC": AminoAcidData(
        three_letter="SEC",
        one_letter="U",
        name="Selenocysteine",
        formula="C3H7NO2Se",
        mw=168.05,
        charge_neutral=0.0,
        charge_ph7=0.0,
        pka_sidechain=5.2,
        is_polar=True,
        is_charged=False,
        is_aromatic=False,
    ),
}

# Add variants and selenium amino acids to main dictionary
AMINO_ACIDS.update(HISTIDINE_VARIANTS)
AMINO_ACIDS.update(SELENIUM_AMINO_ACIDS)


# Residue aliases for common alternative names
RESIDUE_ALIASES: Dict[str, str] = {
    "HSD": "HID",  # CHARMM nomenclature
    "HSE": "HIE",  # CHARMM nomenclature
    "HSP": "HIP",  # CHARMM nomenclature
    "CYX": "CYS",  # Disulfide-bonded cysteine
    "CSS": "CYS",  # Alternative disulfide cysteine
}


# Non-ionizable (neutral) amino acids for convenience
NEUTRAL_AA = frozenset(
    ["ALA", "GLY", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET", "SER", "THR", "ASN", "GLN"]
)


@dataclass(frozen=True)
class PTMData:
    """Data for a post-translational modification.

    Attributes
    ----------
    name : str
        Full name of the modification
    residue_code : str
        3-letter code used in PDB files
    parent_residue : str
        Original amino acid that was modified
    charge_change : float
        Change in charge from parent residue
    mass_change : float
        Change in mass from parent residue (Da)
    formula_change : str
        Description of formula change (e.g., "+HPO3")
    """

    name: str
    residue_code: str
    parent_residue: str
    charge_change: float
    mass_change: float
    formula_change: str


# Post-translational modifications database
PTM_DATABASE: Dict[str, PTMData] = {
    # Phosphorylation (-2 charge at pH 7 due to phosphate group)
    "SEP": PTMData(
        name="Phosphoserine",
        residue_code="SEP",
        parent_residue="SER",
        charge_change=-2.0,
        mass_change=79.97,
        formula_change="+HPO3",
    ),
    "TPO": PTMData(
        name="Phosphothreonine",
        residue_code="TPO",
        parent_residue="THR",
        charge_change=-2.0,
        mass_change=79.97,
        formula_change="+HPO3",
    ),
    "PTR": PTMData(
        name="Phosphotyrosine",
        residue_code="PTR",
        parent_residue="TYR",
        charge_change=-2.0,
        mass_change=79.97,
        formula_change="+HPO3",
    ),
    # Acetylation (neutralizes lysine positive charge)
    "ALY": PTMData(
        name="Acetyllysine",
        residue_code="ALY",
        parent_residue="LYS",
        charge_change=-1.0,
        mass_change=42.04,
        formula_change="+C2H2O",
    ),
    # Methylation (does not change charge)
    "MLY": PTMData(
        name="N-Methyllysine",
        residue_code="MLY",
        parent_residue="LYS",
        charge_change=0.0,
        mass_change=14.03,
        formula_change="+CH2",
    ),
    "M3L": PTMData(
        name="N,N,N-Trimethyllysine",
        residue_code="M3L",
        parent_residue="LYS",
        charge_change=0.0,
        mass_change=42.08,
        formula_change="+C3H6",
    ),
    # Citrullination (removes arginine positive charge)
    "CIR": PTMData(
        name="Citrulline",
        residue_code="CIR",
        parent_residue="ARG",
        charge_change=-1.0,
        mass_change=0.98,
        formula_change="-NH+O",
    ),
    # Oxidation
    "OCS": PTMData(
        name="Cysteinesulfonic acid",
        residue_code="OCS",
        parent_residue="CYS",
        charge_change=-1.0,
        mass_change=47.98,
        formula_change="+O3",
    ),
}


def get_amino_acid(residue_code: str) -> Optional[AminoAcidData]:
    """Get amino acid data by residue code.

    Parameters
    ----------
    residue_code : str
        3-letter residue code (case-insensitive)

    Returns
    -------
    AminoAcidData or None
        Amino acid data if found, None otherwise.

    Examples
    --------
    >>> aa = get_amino_acid("ALA")
    >>> aa.name
    'Alanine'
    >>> aa.mw
    89.09
    """
    code = residue_code.strip().upper()

    # Check aliases first
    if code in RESIDUE_ALIASES:
        code = RESIDUE_ALIASES[code]

    return AMINO_ACIDS.get(code)


def get_neutral_charge(residue_code: str) -> float:
    """Get charge for a residue in neutral (protonated) form.

    Parameters
    ----------
    residue_code : str
        3-letter residue code

    Returns
    -------
    float
        Charge in neutral form (typically 0.0)
    """
    aa = get_amino_acid(residue_code)
    if aa is not None:
        return aa.charge_neutral

    # Check PTMs
    code = residue_code.strip().upper()
    if code in PTM_DATABASE:
        parent = PTM_DATABASE[code].parent_residue
        parent_aa = get_amino_acid(parent)
        if parent_aa:
            return parent_aa.charge_neutral + PTM_DATABASE[code].charge_change

    return 0.0


def get_charge_at_ph74(residue_code: str) -> float:
    """Get charge for a residue at physiological pH 7.4.

    Parameters
    ----------
    residue_code : str
        3-letter residue code

    Returns
    -------
    float
        Charge at pH 7.4

    Examples
    --------
    >>> get_charge_at_ph74("ASP")
    -1.0
    >>> get_charge_at_ph74("LYS")
    1.0
    """
    aa = get_amino_acid(residue_code)
    if aa is not None:
        return aa.charge_ph7

    # Check PTMs
    code = residue_code.strip().upper()
    if code in PTM_DATABASE:
        parent = PTM_DATABASE[code].parent_residue
        parent_aa = get_amino_acid(parent)
        if parent_aa:
            return parent_aa.charge_ph7 + PTM_DATABASE[code].charge_change

    return 0.0


def get_residue_charge_at_ph(residue_code: str, ph: float) -> float:
    """Get charge for a residue at specified pH using Henderson-Hasselbalch.

    Parameters
    ----------
    residue_code : str
        3-letter residue code
    ph : float
        pH value (0-14)

    Returns
    -------
    float
        Charge at specified pH

    Examples
    --------
    >>> get_residue_charge_at_ph("ASP", 7.4)  # doctest: +ELLIPSIS
    -0.999...
    >>> get_residue_charge_at_ph("HIS", 6.0)  # At pKa, 50% protonated
    0.5
    """
    aa = get_amino_acid(residue_code)
    if aa is None:
        # Check PTMs
        code = residue_code.strip().upper()
        if code in PTM_DATABASE:
            ptm = PTM_DATABASE[code]
            parent_charge = get_residue_charge_at_ph(ptm.parent_residue, ph)
            return parent_charge + ptm.charge_change
        return 0.0

    pka = aa.pka_sidechain
    if pka is None:
        return 0.0

    residue = residue_code.strip().upper()

    # Acidic residues: HA ⇌ H+ + A- (lose proton, become negative)
    if residue in ("ASP", "GLU", "CYS", "TYR", "SEC"):
        fraction_deprotonated = 1.0 / (1.0 + 10 ** (pka - ph))
        return -fraction_deprotonated

    # Basic residues: BH+ ⇌ B + H+ (protonated is positive)
    elif residue in ("ARG", "LYS", "HIS", "HID", "HIE", "HIP"):
        fraction_protonated = 1.0 / (1.0 + 10 ** (ph - pka))
        return fraction_protonated

    return 0.0


def get_ptm_charge_adjustment(residue_code: str) -> float:
    """Get charge adjustment for a PTM residue.

    Parameters
    ----------
    residue_code : str
        3-letter residue code for the PTM

    Returns
    -------
    float
        Charge change from parent residue

    Examples
    --------
    >>> get_ptm_charge_adjustment("SEP")  # Phosphoserine
    -2.0
    """
    code = residue_code.strip().upper()
    if code in PTM_DATABASE:
        return PTM_DATABASE[code].charge_change
    return 0.0


def calculate_protein_charge(
    residues: List[str],
    ph: float = 7.4,
    n_terminus: bool = True,
    c_terminus: bool = True,
) -> float:
    """Calculate total charge for a protein or fragment at given pH.

    Parameters
    ----------
    residues : list of str
        List of 3-letter residue codes
    ph : float, optional
        pH for calculation (default: 7.4)
    n_terminus : bool, optional
        Include N-terminal charge (default: True)
    c_terminus : bool, optional
        Include C-terminal charge (default: True)

    Returns
    -------
    float
        Total charge of the protein/fragment

    Examples
    --------
    >>> calculate_protein_charge(["ALA", "ASP", "LYS", "GLY"])  # doctest: +ELLIPSIS
    -0.33...
    """
    from autofragment.data.pka_values import PKA_TERMINAL

    # Sum residue charges
    charge = sum(get_residue_charge_at_ph(res, ph) for res in residues)

    # Terminal charges using Henderson-Hasselbalch
    if n_terminus:
        pka_n = PKA_TERMINAL["N_TERMINUS"]
        # Basic group: protonated = positive
        fraction_protonated = 1.0 / (1.0 + 10 ** (ph - pka_n))
        charge += fraction_protonated

    if c_terminus:
        pka_c = PKA_TERMINAL["C_TERMINUS"]
        # Acidic group: deprotonated = negative
        fraction_deprotonated = 1.0 / (1.0 + 10 ** (pka_c - ph))
        charge -= fraction_deprotonated

    return charge


# Pre-computed charge tables for common pH values
PH74_CHARGES: Dict[str, float] = {code: aa.charge_ph7 for code, aa in AMINO_ACIDS.items()}

PH4_CHARGES: Dict[str, float] = {
    "ASP": -0.31,  # pKa 3.65, ~31% deprotonated at pH 4
    "GLU": -0.05,  # pKa 4.25, ~5% deprotonated at pH 4
    "ARG": +1.0,  # pKa 12.48 >> 4
    "LYS": +1.0,  # pKa 10.53 >> 4
    "HIS": +1.0,  # pKa 6.0, ~99% protonated at pH 4
    "CYS": 0.0,
    "TYR": 0.0,
    **{aa: 0.0 for aa in NEUTRAL_AA},
}

PH10_CHARGES: Dict[str, float] = {
    "ASP": -1.0,
    "GLU": -1.0,
    "ARG": +1.0,  # pKa 12.48, still mostly protonated
    "LYS": +0.25,  # pKa 10.53, ~25% protonated at pH 10
    "HIS": 0.0,  # pKa 6.0, fully deprotonated
    "CYS": -0.4,  # pKa 8.18, ~40% deprotonated
    "TYR": -0.5,  # pKa 10.07, ~50% deprotonated
    **{aa: 0.0 for aa in NEUTRAL_AA},
}
