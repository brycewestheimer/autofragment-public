# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""pKa lookup tables for ionizable groups.

This module provides pKa values for:
- Amino acid side chains
- Amino acid backbone (N/C terminus)
- Nucleic acid phosphate groups

Reference values from biochemistry literature and CRC Handbook.
"""

from __future__ import annotations

from typing import Dict, Optional

# pKa values for ionizable amino acid side chains
# Reference: Stryer, Biochemistry; CRC Handbook of Chemistry and Physics
PKA_SIDECHAIN: Dict[str, float] = {
    # Acidic side chains (lose H+, become negative)
    "ASP": 3.65,  # β-carboxyl
    "GLU": 4.25,  # γ-carboxyl
    "CYS": 8.18,  # Thiol
    "TYR": 10.07,  # Phenolic hydroxyl
    # Basic side chains (gain H+, become positive)
    "HIS": 6.00,  # Imidazole
    "LYS": 10.53,  # ε-amino
    "ARG": 12.48,  # Guanidinium
    # Selenium amino acids
    "SEC": 5.2,  # Selenol (lower than Cys thiol)
}

# pKa values for backbone groups
PKA_TERMINAL: Dict[str, float] = {
    "N_TERMINUS": 7.7,  # α-amino (average, varies by residue)
    "C_TERMINUS": 3.3,  # α-carboxyl (average, varies by residue)
}

# Residue-specific terminal pKa adjustments (deviation from average)
PKA_TERMINAL_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # Glycine has both N and C terminal adjustments
    "GLY": {"N_TERMINUS": -0.2, "C_TERMINUS": -0.1},
    # Proline N-terminal adjustment
    "PRO": {"N_TERMINUS": +0.3},
}

# Nucleic acid pKa values
PKA_NUCLEIC: Dict[str, float] = {
    "PHOSPHATE_PKA1": 1.0,  # First ionization (diester), always deprotonated
    "PHOSPHATE_PKA2": 6.0,  # Second ionization (monoester terminal)
}

# Combined lookup dictionary
PKA_VALUES: Dict[str, float] = {
    # Amino acid sidechains
    "ASP_SIDECHAIN": PKA_SIDECHAIN["ASP"],
    "GLU_SIDECHAIN": PKA_SIDECHAIN["GLU"],
    "CYS_SIDECHAIN": PKA_SIDECHAIN["CYS"],
    "TYR_SIDECHAIN": PKA_SIDECHAIN["TYR"],
    "HIS_SIDECHAIN": PKA_SIDECHAIN["HIS"],
    "LYS_SIDECHAIN": PKA_SIDECHAIN["LYS"],
    "ARG_SIDECHAIN": PKA_SIDECHAIN["ARG"],
    "SEC_SIDECHAIN": PKA_SIDECHAIN["SEC"],
    # Terminals
    "N_TERMINUS": PKA_TERMINAL["N_TERMINUS"],
    "C_TERMINUS": PKA_TERMINAL["C_TERMINUS"],
    # Nucleic acid
    "PHOSPHATE_PKA1": PKA_NUCLEIC["PHOSPHATE_PKA1"],
    "PHOSPHATE_PKA2": PKA_NUCLEIC["PHOSPHATE_PKA2"],
}


def get_pka(group_name: str) -> Optional[float]:
    """Get pKa value for an ionizable group.

    Parameters
    ----------
    group_name : str
        Name of the ionizable group. Supported formats:
        - Residue code alone: "ASP", "HIS", etc.
        - With suffix: "ASP_SIDECHAIN", "N_TERMINUS", etc.

    Returns
    -------
    float or None
        pKa value if found, None otherwise.

    Examples
    --------
    >>> get_pka("ASP")
    3.65
    >>> get_pka("N_TERMINUS")
    7.7
    >>> get_pka("PHOSPHATE_PKA1")
    1.0
    """
    group_upper = group_name.upper()

    # Direct lookup
    if group_upper in PKA_VALUES:
        return PKA_VALUES[group_upper]

    # Try adding _SIDECHAIN suffix
    sidechain_key = f"{group_upper}_SIDECHAIN"
    if sidechain_key in PKA_VALUES:
        return PKA_VALUES[sidechain_key]

    # Try sidechain lookup directly
    if group_upper in PKA_SIDECHAIN:
        return PKA_SIDECHAIN[group_upper]

    return None


def get_terminal_pka(terminal_type: str, residue: Optional[str] = None) -> float:
    """Get pKa for a terminal group, optionally adjusted for specific residue.

    Parameters
    ----------
    terminal_type : str
        Either "N_TERMINUS" or "C_TERMINUS"
    residue : str, optional
        Residue code for specific adjustments

    Returns
    -------
    float
        pKa value, adjusted if residue-specific data available.
    """
    base_pka = PKA_TERMINAL.get(terminal_type, 7.0)

    if residue and residue.upper() in PKA_TERMINAL_ADJUSTMENTS:
        adjustment = PKA_TERMINAL_ADJUSTMENTS[residue.upper()].get(terminal_type, 0.0)
        return base_pka + adjustment

    return base_pka
