# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-level methods and QM/MM partitioning for autofragment.

This module provides support for multi-layer computational schemes
like ONIOM and QM/MM calculations with proper boundary handling.
"""

from autofragment.multilevel.assignment import (
    assign_by_custom,
    assign_by_distance,
    assign_by_element,
    assign_by_residue,
    assign_by_residue_number,
    expand_selection_to_residues,
    validate_layer_assignment,
)
from autofragment.multilevel.layers import (
    ComputationalLayer,
    EmbeddingType,
    LayerType,
    LinkAtom,
    MultiLevelScheme,
)
from autofragment.multilevel.link_atoms import (
    BOND_LENGTHS,
    LinkAtomInfo,
    calculate_g_factor,
    create_link_atoms_for_cut_bonds,
    get_bond_length,
    position_link_atom_fixed_distance,
    position_link_atom_gfactor,
    validate_link_atoms,
)
from autofragment.multilevel.oniom import (
    ONIOMScheme,
    create_oniom_scheme,
    parse_method_basis,
)
from autofragment.multilevel.point_charges import (
    PointCharge,
    PointChargeEmbedding,
    generate_simple_charge_array,
    get_element_charge,
)

__all__ = [
    # Enums
    "LayerType",
    "EmbeddingType",
    # Core classes
    "ComputationalLayer",
    "MultiLevelScheme",
    "LinkAtom",
    # ONIOM
    "ONIOMScheme",
    "create_oniom_scheme",
    "parse_method_basis",
    # Assignment functions
    "assign_by_distance",
    "assign_by_residue",
    "assign_by_residue_number",
    "assign_by_element",
    "assign_by_custom",
    "expand_selection_to_residues",
    "validate_layer_assignment",
    # Advanced link atoms
    "LinkAtomInfo",
    "BOND_LENGTHS",
    "calculate_g_factor",
    "get_bond_length",
    "position_link_atom_gfactor",
    "position_link_atom_fixed_distance",
    "create_link_atoms_for_cut_bonds",
    "validate_link_atoms",
    # Point charges
    "PointCharge",
    "PointChargeEmbedding",
    "get_element_charge",
    "generate_simple_charge_array",
]
