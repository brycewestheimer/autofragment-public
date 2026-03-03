# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Material science data constants and definitions."""

from typing import Dict, Set

# Zeolite Secondary Building Units (SBUs)
# Dictionary mapping SBU name to simplified composition/SMARTS
ZEOLITE_SBUS: Dict[str, str] = {
    "sodalite_cage": "[Si,Al]12O24",  # Beta-cage
    "double_4_ring": "[Si,Al]8O16",   # D4R
    "double_6_ring": "[Si,Al]12O24",  # D6R
    "single_4_ring": "[Si,Al]4O8",    # S4R
    "single_6_ring": "[Si,Al]6O12",   # S6R
    "pentasil_unit": "[Si,Al]12O24",  # MFI unit
}

# Common Polymer Backbones
# SMARTS patterns for common polymer backbones
POLYMER_BACKBONES: Dict[str, str] = {
    "polyethylene": "[CH2]-[CH2]",
    "polypropylene": "[CH2]-[CH](-[CH3])",
    "polystyrene": "[CH2]-[CH](-c1ccccc1)",
    "peg": "[CH2]-[CH2]-[O]",
    "nylon6": "[NH]-[CH2]-[CH2]-[CH2]-[CH2]-[CH2]-[C](=O)",
    "pmma": "[CH2]-[C](-[CH3])(-[C](=O)-[O]-[CH3])",
    "pvc": "[CH2]-[CH](-[Cl])",
}

# Metal definitions
# Common coordination metals in MOFs and porous materials
COORDINATION_METALS: Set[str] = {
    "Zn", "Cu", "Fe", "Co", "Ni", "Mn", "Cr", "Zr",
    "Ti", "V", "Al", "Mg", "Cd", "Pb", "Hf", "Mo", "W"
}

# Perovskite A and B site common elements
PEROVSKITE_A_SITE: Set[str] = {
    "Sr", "Ba", "Ca", "La", "K", "Na", "Cs", "Rb", "Pb", "Bi"
}

PEROVSKITE_B_SITE: Set[str] = {
    "Ti", "Nb", "Ta", "Zr", "Mn", "Fe", "Co", "Ni", "Sn", "Hf"
}

# Framework topologies (3-letter codes)
FRAMEWORK_CODES: Set[str] = {
    "MFI", "FAU", "BEA", "LTA", "CHA", "MOR", "ZIF", "MOF"
}
