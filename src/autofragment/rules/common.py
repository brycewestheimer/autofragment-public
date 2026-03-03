# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Common chemical fragmentation rules.

This module provides universally applicable chemical fragmentation rules
that preserve fundamental chemical structures like aromatic rings,
multiple bonds, metal coordination, and functional groups.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

from .base import FragmentationRule, RuleAction

if TYPE_CHECKING:
    from autofragment.core.types import ChemicalSystem


class AromaticRingRule(FragmentationRule):
    """Never break bonds within aromatic ring systems.

    This rule identifies aromatic bonds using bond type markers or
    ring membership with aromatic characteristics, and marks them as
    MUST_NOT_BREAK.

    Aromatic bonds are identified by:
    1. Bond type attribute set to "aromatic"
    2. Bond order of 1.5 (typical aromatic bond order)
    3. Membership in a ring with aromatic characteristics

    Attributes:
        name: Rule identifier ("aromatic_ring")

    Example:
        >>> rule = AromaticRingRule()
        >>> rule.applies_to(benzene_c_c_bond, system)
        True
        >>> rule.action()
        RuleAction.MUST_NOT_BREAK
    """

    name = "aromatic_ring"

    # Elements commonly found in aromatic rings
    AROMATIC_ELEMENTS = {"C", "N", "O", "S"}

    def __init__(self, priority: Optional[int] = None):
        """Initialize the aromatic ring rule.

        Args:
            priority: Rule priority. Defaults to PRIORITY_CRITICAL.
        """
        super().__init__(priority=priority or self.PRIORITY_CRITICAL)

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is part of an aromatic ring.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if the bond is aromatic, False otherwise.
        """
        graph = system.to_graph()

        # Check bond attributes for aromaticity
        bond_data = graph.get_bond(bond[0], bond[1])
        if bond_data:
            # Check explicit aromatic marker
            if bond_data.get("bond_type") == "aromatic":
                return True
            if bond_data.get("aromatic", False):
                return True
            # Check bond order (aromatic = 1.5)
            if bond_data.get("order", 1.0) == 1.5:
                return True

        # Check if both atoms are in the same ring and are aromatic elements
        atom1_data = graph.get_atom(bond[0])
        atom2_data = graph.get_atom(bond[1])

        if (atom1_data["element"] in self.AROMATIC_ELEMENTS and
            atom2_data["element"] in self.AROMATIC_ELEMENTS):
            # Check if bond is in a ring (5 or 6 membered, typical for aromatics)
            if graph.is_in_ring(bond[0], bond[1]):
                rings = graph.find_rings()
                for ring in rings:
                    if bond[0] in ring and bond[1] in ring:
                        # Aromatic rings are typically 5 or 6 membered
                        if len(ring) in (5, 6):
                            return True

        return False

    def action(self) -> RuleAction:
        """Aromatic bonds must never be broken.

        Returns:
            RuleAction.MUST_NOT_BREAK
        """
        return RuleAction.MUST_NOT_BREAK


class DoubleBondRule(FragmentationRule):
    """Never break double or triple bonds.

    Identifies bonds with order > 1.0 and marks them as MUST_NOT_BREAK.
    This preserves the structural integrity of alkenes, alkynes, carbonyls,
    imines, and other multiply-bonded systems.

    Attributes:
        name: Rule identifier ("double_bond")
        min_order: Minimum bond order to trigger the rule (default 1.5)

    Example:
        >>> rule = DoubleBondRule()
        >>> rule.applies_to(carbonyl_bond, system)  # C=O
        True
        >>> rule.action()
        RuleAction.MUST_NOT_BREAK
    """

    name = "double_bond"

    def __init__(
        self,
        min_order: float = 1.5,
        priority: Optional[int] = None
    ):
        """Initialize the double bond rule.

        Args:
            min_order: Minimum bond order to protect. Default 1.5 catches
                      aromatic (1.5), double (2.0), and triple (3.0) bonds.
            priority: Rule priority. Defaults to PRIORITY_CRITICAL.
        """
        super().__init__(priority=priority or self.PRIORITY_CRITICAL)
        self.min_order = min_order

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond order >= min_order.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if bond order is at or above the threshold.
        """
        graph = system.to_graph()
        bond_data = graph.get_bond(bond[0], bond[1])
        if bond_data is None:
            return False
        return bond_data.get("order", 1.0) >= self.min_order

    def action(self) -> RuleAction:
        """Multiple bonds must never be broken.

        Returns:
            RuleAction.MUST_NOT_BREAK
        """
        return RuleAction.MUST_NOT_BREAK


class MetalCoordinationRule(FragmentationRule):
    """Preserve metal-ligand coordination bonds.

    Identifies bonds involving transition metals (or specified metals)
    and marks them with a configurable action (default MUST_NOT_BREAK).

    This is essential for metalloenzymes, organometallic compounds,
    and metal-organic frameworks.

    Attributes:
        name: Rule identifier ("metal_coordination")
        metals: Set of metal element symbols to match

    Example:
        >>> rule = MetalCoordinationRule()
        >>> rule.applies_to(fe_nitrogen_bond, heme_system)
        True
    """

    name = "metal_coordination"

    # Default transition metals and common coordination metals
    DEFAULT_METALS: Set[str] = {
        "Fe", "Co", "Ni", "Cu", "Zn", "Mn", "Cr", "Mo", "W",
        "Ru", "Rh", "Pd", "Pt", "Au", "Ag", "Ti", "V", "Ir",
        "Os", "Re", "Tc", "Nb", "Ta", "Hf", "Zr", "Y", "Sc",
        # Lanthanides commonly used
        "La", "Ce", "Nd", "Eu", "Gd", "Tb", "Dy", "Er", "Yb",
        # Main group metals that coordinate
        "Al", "Ga", "In", "Sn", "Pb", "Bi", "Mg", "Ca", "Sr", "Ba"
    }

    def __init__(
        self,
        metals: Optional[Set[str]] = None,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        priority: Optional[int] = None
    ):
        """Initialize the metal coordination rule.

        Args:
            metals: Set of metal element symbols. Uses DEFAULT_METALS if None.
            rule_action: Action to apply (MUST_NOT_BREAK or PREFER_KEEP).
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self.metals = metals if metals is not None else self.DEFAULT_METALS.copy()
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond involves a metal atom.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if either atom in the bond is a tracked metal.
        """
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])
        return atom1["element"] in self.metals or atom2["element"] in self.metals

    def action(self) -> RuleAction:
        """Return the configured action for metal coordination bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class FunctionalGroupRule(FragmentationRule):
    """Keep common functional groups intact.

    Identifies bonds within defined functional groups using local
    environment analysis and marks them as MUST_NOT_BREAK.

    Default protected functional groups include:
    - Carboxyl (-COOH)
    - Carbonyl (C=O)
    - Amino (-NH2, -NHR, -NR2)
    - Hydroxyl (-OH)
    - Nitro (-NO2)
    - Sulfhydryl (-SH)
    - Phosphate (-PO4)
    - Sulfonate (-SO3)

    Attributes:
        name: Rule identifier ("functional_group")

    Example:
        >>> rule = FunctionalGroupRule()
        >>> rule.applies_to(carboxyl_c_o_bond, system)
        True
    """

    name = "functional_group"

    # Functional groups defined as central atom and its expected neighbors
    # Format: {group_name: {"center": element, "pattern": [(neighbor_element, min_count), ...]}}
    DEFAULT_GROUPS: Dict[str, Dict[str, Any]] = {
        "carboxyl": {"center": "C", "neighbors": {"O": 2}},
        "nitro": {"center": "N", "neighbors": {"O": 2}},
        "sulfonate": {"center": "S", "neighbors": {"O": 3}},
        "phosphate": {"center": "P", "neighbors": {"O": 3}},
        "amino": {"center": "N", "neighbors": {"H": 1}},  # At least one H
    }

    def __init__(
        self,
        groups: Optional[Dict[str, Dict[str, Any]]] = None,
        priority: Optional[int] = None
    ):
        """Initialize the functional group rule.

        Args:
            groups: Custom functional group definitions. Uses DEFAULT_GROUPS if None.
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self.groups = groups if groups is not None else self.DEFAULT_GROUPS.copy()
        # Cache for detected functional group atoms
        self._fg_cache: Dict[int, Set[Tuple[int, int]]] = {}

    def _find_functional_groups(self, system: "ChemicalSystem") -> Set[Tuple[int, int]]:
        """Find all bonds that are part of functional groups.

        Args:
            system: ChemicalSystem to analyze.

        Returns:
            Set of bond tuples that are within functional groups.
        """
        system_id = id(system)
        if system_id in self._fg_cache:
            return self._fg_cache[system_id]

        protected_bonds: Set[Tuple[int, int]] = set()

        # For each atom, check if it's a potential functional group center
        for i, atom in enumerate(system.atoms):
            for group_name, pattern in self.groups.items():
                if atom.symbol == pattern["center"]:
                    # Count neighbors by element
                    neighbor_counts: Dict[str, int] = {}
                    neighbor_indices: list = []

                    for bond_info in system.bonds:
                        if bond_info["atom1"] == i:
                            neighbor_idx = bond_info["atom2"]
                        elif bond_info["atom2"] == i:
                            neighbor_idx = bond_info["atom1"]
                        else:
                            continue

                        neighbor_elem = system.atoms[neighbor_idx].symbol
                        neighbor_counts[neighbor_elem] = neighbor_counts.get(neighbor_elem, 0) + 1
                        neighbor_indices.append(neighbor_idx)

                    # Check if pattern matches
                    matches = True
                    for elem, min_count in pattern["neighbors"].items():
                        if neighbor_counts.get(elem, 0) < min_count:
                            matches = False
                            break

                    if matches:
                        # Add all bonds from center to neighbors
                        for neighbor_idx in neighbor_indices:
                            bond = tuple(sorted([i, neighbor_idx]))
                            protected_bonds.add(bond)

        self._fg_cache[system_id] = protected_bonds
        return protected_bonds

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is within a functional group.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if the bond is part of a protected functional group.
        """
        protected = self._find_functional_groups(system)
        normalized_bond = tuple(sorted(bond))
        return normalized_bond in protected

    def action(self) -> RuleAction:
        """Functional group bonds should not be broken.

        Returns:
            RuleAction.MUST_NOT_BREAK
        """
        return RuleAction.MUST_NOT_BREAK
