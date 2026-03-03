# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Biological and chemical data module.

This module provides comprehensive data tables for:
- Amino acid properties and charges at various pH levels
- Nucleotide definitions for DNA and RNA
- pKa lookup tables for ionizable groups
- Post-translational modification (PTM) data
"""

from autofragment.data.amino_acids import (
    AMINO_ACIDS,
    PTM_DATABASE,
    AminoAcidData,
    PTMData,
    calculate_protein_charge,
    get_amino_acid,
    get_charge_at_ph74,
    get_neutral_charge,
    get_ptm_charge_adjustment,
    get_residue_charge_at_ph,
)
from autofragment.data.nucleotides import (
    DNA_NUCLEOTIDES,
    NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    NucleotideData,
    get_nucleic_acid_charge,
    get_nucleotide,
)
from autofragment.data.pka_values import (
    PKA_VALUES,
    get_pka,
)

__all__ = [
    # Amino acids
    "AMINO_ACIDS",
    "AminoAcidData",
    "get_amino_acid",
    "get_residue_charge_at_ph",
    "get_charge_at_ph74",
    "get_neutral_charge",
    "calculate_protein_charge",
    "PTM_DATABASE",
    "PTMData",
    "get_ptm_charge_adjustment",
    # Nucleotides
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "NUCLEOTIDES",
    "NucleotideData",
    "get_nucleotide",
    "get_nucleic_acid_charge",
    # pKa
    "PKA_VALUES",
    "get_pka",
]
