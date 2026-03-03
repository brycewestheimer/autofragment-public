# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Nucleotide definitions for DNA and RNA.

This module provides comprehensive data for DNA and RNA nucleotides including:
- Nucleotide properties (formula, molecular weight)
- Charge breakdown (backbone, base, sugar)
- Support for both standard and modified nucleotides

Reference values from nucleic acid chemistry literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NucleotideData:
    """Data for a nucleotide.

    Attributes
    ----------
    code : str
        PDB residue code (e.g., "DA", "A")
    one_letter : str
        One-letter base code (A, T, G, C, U)
    name : str
        Full name (e.g., "Deoxyadenosine")
    nucleic_type : str
        "DNA" or "RNA"
    base : str
        Base type (A, T, G, C, U)
    formula : str
        Molecular formula
    mw : float
        Molecular weight in Da
    charge_backbone : float
        Phosphate backbone charge (typically -1)
    charge_base : float
        Nucleobase charge (typically 0)
    charge_sugar : float
        Sugar moiety charge (typically 0)
    """

    code: str
    one_letter: str
    name: str
    nucleic_type: str
    base: str
    formula: str
    mw: float
    charge_backbone: float
    charge_base: float
    charge_sugar: float

    @property
    def total_charge(self) -> float:
        """Total charge of the nucleotide."""
        return self.charge_backbone + self.charge_base + self.charge_sugar


# DNA nucleotides (deoxyribonucleotides)
DNA_NUCLEOTIDES: Dict[str, NucleotideData] = {
    "DA": NucleotideData(
        code="DA",
        one_letter="A",
        name="Deoxyadenosine 5'-monophosphate",
        nucleic_type="DNA",
        base="A",
        formula="C10H14N5O6P",
        mw=331.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "DT": NucleotideData(
        code="DT",
        one_letter="T",
        name="Thymidine 5'-monophosphate",
        nucleic_type="DNA",
        base="T",
        formula="C10H15N2O8P",
        mw=322.21,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "DG": NucleotideData(
        code="DG",
        one_letter="G",
        name="Deoxyguanosine 5'-monophosphate",
        nucleic_type="DNA",
        base="G",
        formula="C10H14N5O7P",
        mw=347.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "DC": NucleotideData(
        code="DC",
        one_letter="C",
        name="Deoxycytidine 5'-monophosphate",
        nucleic_type="DNA",
        base="C",
        formula="C9H14N3O7P",
        mw=307.20,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
}

# RNA nucleotides (ribonucleotides)
RNA_NUCLEOTIDES: Dict[str, NucleotideData] = {
    "A": NucleotideData(
        code="A",
        one_letter="A",
        name="Adenosine 5'-monophosphate",
        nucleic_type="RNA",
        base="A",
        formula="C10H14N5O7P",
        mw=347.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "U": NucleotideData(
        code="U",
        one_letter="U",
        name="Uridine 5'-monophosphate",
        nucleic_type="RNA",
        base="U",
        formula="C9H13N2O9P",
        mw=324.18,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "G": NucleotideData(
        code="G",
        one_letter="G",
        name="Guanosine 5'-monophosphate",
        nucleic_type="RNA",
        base="G",
        formula="C10H14N5O8P",
        mw=363.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "C": NucleotideData(
        code="C",
        one_letter="C",
        name="Cytidine 5'-monophosphate",
        nucleic_type="RNA",
        base="C",
        formula="C9H14N3O8P",
        mw=323.20,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
}

# Alternative PDB codes for nucleotides
NUCLEOTIDE_ALIASES: Dict[str, str] = {
    # DNA alternate codes (3-letter to standard 2-letter)
    "ADE": "DA",  # Adenine in some PDBs maps to DNA by default
    "THY": "DT",
    "GUA": "DG",  # Guanine in some PDBs maps to DNA by default
    "CYT": "DC",  # Cytosine in some PDBs maps to DNA by default
    # RNA alternate codes
    "URA": "U",
    "RADE": "A",  # RNA adenine (explicit)
    "RGUA": "G",  # RNA guanine (explicit)
    "RCYT": "C",  # RNA cytosine (explicit)
    # Terminal nucleotide variants commonly found in PDB
    "DA5": "DA",  # 5' terminal
    "DA3": "DA",  # 3' terminal
    "DT5": "DT",
    "DT3": "DT",
    "DG5": "DG",
    "DG3": "DG",
    "DC5": "DC",
    "DC3": "DC",
    "A5": "A",
    "A3": "A",
    "U5": "U",
    "U3": "U",
    "G5": "G",
    "G3": "G",
    "C5": "C",
    "C3": "C",
}

# Combined dictionary of all nucleotides
NUCLEOTIDES: Dict[str, NucleotideData] = {
    **DNA_NUCLEOTIDES,
    **RNA_NUCLEOTIDES,
}

# Modified nucleotides (common in structural biology)
MODIFIED_NUCLEOTIDES: Dict[str, NucleotideData] = {
    # Modified DNA
    "5CM": NucleotideData(
        code="5CM",
        one_letter="C",
        name="5-Methylcytidine (DNA)",
        nucleic_type="DNA",
        base="C",
        formula="C10H16N3O7P",
        mw=321.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    # Modified RNA
    "PSU": NucleotideData(
        code="PSU",
        one_letter="U",
        name="Pseudouridine",
        nucleic_type="RNA",
        base="U",
        formula="C9H13N2O9P",
        mw=324.18,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "OMC": NucleotideData(
        code="OMC",
        one_letter="C",
        name="2'-O-methylcytidine",
        nucleic_type="RNA",
        base="C",
        formula="C10H16N3O8P",
        mw=337.22,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
    "OMG": NucleotideData(
        code="OMG",
        one_letter="G",
        name="2'-O-methylguanosine",
        nucleic_type="RNA",
        base="G",
        formula="C11H16N5O8P",
        mw=377.24,
        charge_backbone=-1.0,
        charge_base=0.0,
        charge_sugar=0.0,
    ),
}

NUCLEOTIDES.update(MODIFIED_NUCLEOTIDES)


# Phosphate pKa values
PHOSPHATE_PKA = {
    "pKa1": 1.0,  # First ionization (diester), always deprotonated at pH > 3
    "pKa2": 6.0,  # Second ionization (terminal monoester)
}


def get_nucleotide(code: str) -> Optional[NucleotideData]:
    """Get nucleotide data by residue code.

    Parameters
    ----------
    code : str
        Nucleotide residue code (case-insensitive)

    Returns
    -------
    NucleotideData or None
        Nucleotide data if found, None otherwise.

    Examples
    --------
    >>> nuc = get_nucleotide("DA")
    >>> nuc.name
    "Deoxyadenosine 5'-monophosphate"
    >>> nuc.total_charge
    -1.0
    """
    code_upper = code.strip().upper()

    # Check aliases first
    if code_upper in NUCLEOTIDE_ALIASES:
        code_upper = NUCLEOTIDE_ALIASES[code_upper]

    return NUCLEOTIDES.get(code_upper)


def is_dna_nucleotide(code: str) -> bool:
    """Check if a residue code is a DNA nucleotide.

    Parameters
    ----------
    code : str
        Residue code to check

    Returns
    -------
    bool
        True if DNA nucleotide
    """
    nuc = get_nucleotide(code)
    return nuc is not None and nuc.nucleic_type == "DNA"


def is_rna_nucleotide(code: str) -> bool:
    """Check if a residue code is an RNA nucleotide.

    Parameters
    ----------
    code : str
        Residue code to check

    Returns
    -------
    bool
        True if RNA nucleotide
    """
    nuc = get_nucleotide(code)
    return nuc is not None and nuc.nucleic_type == "RNA"


def get_nucleic_acid_charge(
    nucleotides: List[str],
    include_5prime_phosphate: bool = False,
    include_3prime_phosphate: bool = False,
) -> float:
    """Calculate total charge for a nucleic acid chain.

    At physiological pH, each phosphate group carries -1 charge.
    The phosphate is the bridging group between nucleotides.

    For an internal nucleotide, we count:
    - 1 phosphate per nucleotide (contributes to chain)

    Termini:
    - 5' end: typically has phosphate (-1) or hydroxyl (0)
    - 3' end: typically has hydroxyl (0) or phosphate (-1)

    Parameters
    ----------
    nucleotides : list of str
        List of nucleotide codes
    include_5prime_phosphate : bool, optional
        Whether 5' terminal phosphate is present (default: False)
    include_3prime_phosphate : bool, optional
        Whether 3' terminal phosphate is present (default: False)

    Returns
    -------
    float
        Total charge of the nucleic acid chain

    Examples
    --------
    >>> get_nucleic_acid_charge(["DA", "DT", "DG", "DC"])
    -3.0
    >>> get_nucleic_acid_charge(["DA", "DT", "DG", "DC"], include_5prime_phosphate=True)
    -4.0
    """
    n = len(nucleotides)
    if n == 0:
        return 0.0

    # Each internal phosphate bridge contributes -1
    # For n nucleotides, there are (n-1) internal phosphate bridges
    internal_phosphates = n - 1

    # Terminal phosphates
    terminal_phosphates = 0
    if include_5prime_phosphate:
        terminal_phosphates += 1
    if include_3prime_phosphate:
        terminal_phosphates += 1

    total_charge: float = float(-(internal_phosphates + terminal_phosphates))

    # Add any base charge contributions (typically 0 at neutral pH)
    for code in nucleotides:
        nuc = get_nucleotide(code)
        if nuc is not None:
            total_charge += nuc.charge_base

    return float(total_charge)


def get_phosphate_charge_at_ph(ph: float, is_terminal: bool = False) -> float:
    """Get phosphate charge at a given pH.

    Internal phosphodiester groups have pKa ~1 and are fully ionized
    at any biologically relevant pH.

    Terminal phosphomonoester groups have:
    - pKa1 ~1 (always ionized)
    - pKa2 ~6 (partially ionized at neutral pH)

    Parameters
    ----------
    ph : float
        pH value
    is_terminal : bool, optional
        Whether this is a terminal monoester phosphate

    Returns
    -------
    float
        Phosphate charge (-1 to -2)
    """
    if not is_terminal:
        # Internal diester, single ionization
        return -1.0

    # Terminal monoester, two ionizations
    pka1 = PHOSPHATE_PKA["pKa1"]
    pka2 = PHOSPHATE_PKA["pKa2"]

    # First ionization (always deprotonated at biological pH)
    charge1 = -1.0 / (1.0 + 10 ** (pka1 - ph))

    # Second ionization
    charge2 = -1.0 / (1.0 + 10 ** (pka2 - ph))

    return charge1 + charge2


# Watson-Crick base pairing rules
WATSON_CRICK_PAIRS = {
    # DNA pairs
    ("A", "T"): "Watson-Crick",
    ("T", "A"): "Watson-Crick",
    ("G", "C"): "Watson-Crick",
    ("C", "G"): "Watson-Crick",
    # RNA pairs
    ("A", "U"): "Watson-Crick",
    ("U", "A"): "Watson-Crick",
    # Wobble pair (common in tRNA)
    ("G", "U"): "Wobble",
    ("U", "G"): "Wobble",
}


def can_base_pair(base1: str, base2: str) -> Optional[str]:
    """Check if two bases can form a canonical base pair.

    Parameters
    ----------
    base1, base2 : str
        Single-letter base codes

    Returns
    -------
    str or None
        Base pair type if they can pair, None otherwise
    """
    return WATSON_CRICK_PAIRS.get((base1.upper(), base2.upper()))
