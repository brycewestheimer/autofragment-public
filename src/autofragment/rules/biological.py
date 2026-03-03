# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Biological fragmentation rules for proteins and nucleic acids.

This module provides rules specifically designed for protein and DNA/RNA
systems, with configurable actions per user requirements.

Key design principle: Biological rules are CONFIGURABLE, not absolute,
allowing users to choose appropriate fragmentation strategies for their
specific use case (FMO, residue-based, domain-based, etc.).
"""
import math
from typing import TYPE_CHECKING, Optional, Set, Tuple

from .base import FragmentationRule, RuleAction

if TYPE_CHECKING:
    from autofragment.core.types import ChemicalSystem


class PeptideBondRule(FragmentationRule):
    """Configurable peptide bond handling.

    **Per user requirements, this rule is CONFIGURABLE.**

    Peptide bonds (C-N between amino acid residues) can be:
    - MUST_NOT_BREAK: Never break (traditional FMO)
    - PREFER_KEEP: Try to keep, but can break if needed (default)
    - ALLOW: No preference (neutral)
    - PREFER_BREAK: Good fragmentation points (alpha carbon approach)

    The rule identifies peptide bonds by checking for the characteristic
    backbone pattern: N-C(=O)-C bond where the C-N bond is the peptide bond.

    Attributes:
        name: Rule identifier ("peptide_bond")

    Example:
        >>> # Traditional: keep peptide bonds
        >>> rule = PeptideBondRule(action=RuleAction.PREFER_KEEP)

        >>> # For residue-based fragmentation: break at peptide bonds
        >>> rule = PeptideBondRule(action=RuleAction.PREFER_BREAK)
    """

    name = "peptide_bond"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.PREFER_KEEP,
        priority: Optional[int] = None
    ):
        """Initialize the peptide bond rule.

        Args:
            rule_action: Action to apply to peptide bonds. Default is PREFER_KEEP.
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is a peptide bond (backbone C-N).

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if this is a backbone peptide bond.
        """
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])

        # Check for C-N bond
        elements = {atom1["element"], atom2["element"]}
        if elements != {"C", "N"}:
            return False

        # Identify which is C and which is N
        if atom1["element"] == "C":
            c_idx, n_idx = bond[0], bond[1]
        else:
            c_idx, n_idx = bond[1], bond[0]

        # Check backbone context using bond pattern
        return self._is_backbone_peptide(graph, c_idx, n_idx, system)

    def _is_backbone_peptide(self, graph, c_idx: int, n_idx: int, system: "ChemicalSystem") -> bool:
        """Verify this is a backbone peptide, not a sidechain amide.

        A backbone peptide bond has:
        - C is a carbonyl carbon (bonded to O with order >= 2 or marked as C=O)
        - C is also bonded to another carbon (C-alpha of that residue)
        - N is bonded to another carbon (C-alpha of next residue)
        """
        # Get neighbors of C (the carbonyl carbon)
        c_has_carbonyl_o = False
        c_has_alpha_c = False

        for bond_info in system.bonds:
            if bond_info["atom1"] == c_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == c_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if neighbor_idx == n_idx:
                continue  # Skip the N we're already considering

            neighbor = system.atoms[neighbor_idx]

            # Check for carbonyl oxygen
            if neighbor.symbol == "O":
                order = bond_info.get("order", 1.0)
                if order >= 1.5:  # Double bond or aromatic
                    c_has_carbonyl_o = True

            # Check for alpha carbon connection
            if neighbor.symbol == "C":
                c_has_alpha_c = True

        # Get neighbors of N
        n_has_alpha_c = False
        for bond_info in system.bonds:
            if bond_info["atom1"] == n_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == n_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if neighbor_idx == c_idx:
                continue  # Skip the C we're already considering

            neighbor = system.atoms[neighbor_idx]
            if neighbor.symbol == "C":
                n_has_alpha_c = True

        return c_has_carbonyl_o and c_has_alpha_c and n_has_alpha_c

    def action(self) -> RuleAction:
        """Return the configured action for peptide bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class DisulfideBondRule(FragmentationRule):
    """Configurable disulfide bridge handling.

    **Per user requirements, this rule is CONFIGURABLE.**

    Disulfide bonds (S-S between cysteine residues) typically hold
    protein structure together but may need to be broken for
    certain fragmentation schemes.

    Attributes:
        name: Rule identifier ("disulfide_bond")

    Example:
        >>> # Preserve disulfides (keep structure)
        >>> rule = DisulfideBondRule(action=RuleAction.MUST_NOT_BREAK)

        >>> # Allow breaking for domain separation
        >>> rule = DisulfideBondRule(action=RuleAction.ALLOW)
    """

    name = "disulfide_bond"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.PREFER_KEEP,
        priority: Optional[int] = None
    ):
        """Initialize the disulfide bond rule.

        Args:
            rule_action: Action to apply to disulfide bonds. Default is PREFER_KEEP.
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is a disulfide bridge (S-S).

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if this is a disulfide bond.
        """
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])

        # Both atoms must be sulfur
        if atom1["element"] != "S" or atom2["element"] != "S":
            return False

        # Verify both sulfurs are likely from cysteine residues
        # (each S should be bonded to a carbon, which is the C-beta of Cys)
        return (self._is_cysteine_sulfur(bond[0], system) and
                self._is_cysteine_sulfur(bond[1], system))

    def _is_cysteine_sulfur(self, s_idx: int, system: "ChemicalSystem") -> bool:
        """Check if this sulfur is likely from a cysteine residue.

        Cysteine sulfur is bonded to a single carbon (C-beta).
        """
        carbon_neighbors = 0
        for bond_info in system.bonds:
            if bond_info["atom1"] == s_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == s_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if system.atoms[neighbor_idx].symbol == "C":
                carbon_neighbors += 1

        # Cysteine S has exactly 1 C neighbor (the C-beta)
        # In a disulfide, it also has 1 S neighbor
        return carbon_neighbors == 1

    def action(self) -> RuleAction:
        """Return the configured action for disulfide bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class AlphaBetaCarbonRule(FragmentationRule):
    """Mark alpha-beta carbon bonds as preferred break points.

    The bond between C-alpha and C-beta (sidechain attachment)
    is often a good fragmentation point for residue-based schemes.

    This rule identifies C-alpha by its backbone context:
    - Bonded to N (amide nitrogen from peptide bond)
    - Bonded to C (carbonyl carbon of peptide bond)
    - Bonded to H (alpha hydrogen, except in proline)
    - Bonded to C-beta (sidechain, except glycine)

    Attributes:
        name: Rule identifier ("alpha_beta_carbon")

    Example:
        >>> rule = AlphaBetaCarbonRule()  # Default: PREFER_BREAK
        >>> rule.applies_to(ca_cb_bond, system)
        True
    """

    name = "alpha_beta_carbon"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.PREFER_BREAK,
        priority: Optional[int] = None
    ):
        """Initialize the alpha-beta carbon rule.

        Args:
            rule_action: Action to apply. Default is PREFER_BREAK.
            priority: Rule priority. Defaults to PRIORITY_MEDIUM.
        """
        super().__init__(priority=priority or self.PRIORITY_MEDIUM)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is C-alpha to C-beta.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if this is a C-alpha to C-beta bond.
        """
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])

        # Both must be carbons
        if atom1["element"] != "C" or atom2["element"] != "C":
            return False

        # Check if one is C-alpha (has backbone context) and other is C-beta
        if self._is_alpha_carbon(bond[0], system):
            return self._is_beta_carbon(bond[1], bond[0], system)
        elif self._is_alpha_carbon(bond[1], system):
            return self._is_beta_carbon(bond[0], bond[1], system)

        return False

    def _is_alpha_carbon(self, c_idx: int, system: "ChemicalSystem") -> bool:
        """Check if this carbon has backbone (alpha) characteristics.

        C-alpha is bonded to:
        - At least one N (backbone amide)
        - At least one C that's a carbonyl carbon
        """
        has_n_neighbor = False
        has_carbonyl_c_neighbor = False

        for bond_info in system.bonds:
            if bond_info["atom1"] == c_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == c_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            neighbor = system.atoms[neighbor_idx]

            if neighbor.symbol == "N":
                has_n_neighbor = True
            elif neighbor.symbol == "C":
                # Check if this C is a carbonyl carbon (has C=O)
                if self._has_carbonyl_oxygen(neighbor_idx, system):
                    has_carbonyl_c_neighbor = True

        return has_n_neighbor and has_carbonyl_c_neighbor

    def _has_carbonyl_oxygen(self, c_idx: int, system: "ChemicalSystem") -> bool:
        """Check if carbon has a carbonyl oxygen attached."""
        for bond_info in system.bonds:
            if bond_info["atom1"] == c_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == c_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            neighbor = system.atoms[neighbor_idx]
            if neighbor.symbol == "O":
                order = bond_info.get("order", 1.0)
                if order >= 1.5:
                    return True
        return False

    def _is_beta_carbon(self, cb_idx: int, ca_idx: int, system: "ChemicalSystem") -> bool:
        """Check if this is a C-beta (sidechain carbon attached to C-alpha).

        C-beta should not have backbone characteristics itself.
        """
        # C-beta should NOT be attached to backbone N
        for bond_info in system.bonds:
            if bond_info["atom1"] == cb_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == cb_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if neighbor_idx == ca_idx:
                continue

            neighbor = system.atoms[neighbor_idx]
            # C-beta shouldn't be bonded to N (that would make it backbone)
            if neighbor.symbol == "N":
                return False

        return True

    def action(self) -> RuleAction:
        """Return the configured action for C-alpha to C-beta bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class ProlineRingRule(FragmentationRule):
    """Never break bonds within proline's pyrrolidine ring.

    Proline has a unique cyclic sidechain connected to backbone N.
    The 5-membered pyrrolidine ring bonds must never be broken to
    maintain the amino acid's structural integrity.

    Attributes:
        name: Rule identifier ("proline_ring")

    Example:
        >>> rule = ProlineRingRule()
        >>> rule.action()
        RuleAction.MUST_NOT_BREAK
    """

    name = "proline_ring"

    def __init__(self, priority: Optional[int] = None):
        """Initialize the proline ring rule.

        Args:
            priority: Rule priority. Defaults to PRIORITY_CRITICAL.
        """
        super().__init__(priority=priority or self.PRIORITY_CRITICAL)

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is part of proline's pyrrolidine ring.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if this bond is in a proline-like ring.
        """
        graph = system.to_graph()

        # Check if bond is in any ring
        if not graph.is_in_ring(bond[0], bond[1]):
            return False

        # Look for 5-membered rings containing exactly one N
        rings = graph.find_rings()
        for ring in rings:
            if len(ring) != 5:
                continue

            if bond[0] not in ring or bond[1] not in ring:
                continue

            # Count nitrogen atoms in ring
            n_count = sum(1 for idx in ring if system.atoms[idx].symbol == "N")
            c_count = sum(1 for idx in ring if system.atoms[idx].symbol == "C")

            # Proline ring: 1 N, 4 C
            if n_count == 1 and c_count == 4:
                return True

        return False

    def action(self) -> RuleAction:
        """Proline ring bonds must never be broken.

        Returns:
            RuleAction.MUST_NOT_BREAK
        """
        return RuleAction.MUST_NOT_BREAK


class HydrogenBondRule(FragmentationRule):
    """Configurable hydrogen bond handling.

    **Per user requirements, this rule is CONFIGURABLE.**

    Important: For water clusters and many systems, hydrogen bonds
    MUST be broken to create fragments. Default action is ALLOW.

    Note: Hydrogen bonds are typically not represented as covalent bonds
    in molecular graphs. This rule handles non-bonded H-bond interactions
    that may be tracked separately in the system.

    Attributes:
        name: Rule identifier ("hydrogen_bond")
        distance_cutoff: Maximum D...A distance in Angstroms
        angle_cutoff: Minimum D-H...A angle in degrees

    Example:
        >>> # Allow breaking (default for water clusters)
        >>> rule = HydrogenBondRule(action=RuleAction.ALLOW)

        >>> # Prefer keeping (for secondary structure)
        >>> rule = HydrogenBondRule(action=RuleAction.PREFER_KEEP)
    """

    name = "hydrogen_bond"

    # Common H-bond donor elements
    DONOR_ELEMENTS: Set[str] = {"N", "O", "F"}
    # Common H-bond acceptor elements
    ACCEPTOR_ELEMENTS: Set[str] = {"N", "O", "F", "S"}

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.ALLOW,  # Note: ALLOW by default!
        distance_cutoff: float = 3.5,  # Angstroms
        angle_cutoff: float = 120.0,   # degrees
        priority: Optional[int] = None
    ):
        """Initialize the hydrogen bond rule.

        Args:
            rule_action: Action to apply. Default is ALLOW (not restrictive).
            distance_cutoff: Maximum donor-acceptor distance in Angstroms.
            angle_cutoff: Minimum D-H...A angle in degrees.
            priority: Rule priority. Defaults to PRIORITY_LOW.
        """
        super().__init__(priority=priority or self.PRIORITY_LOW)
        self._action = rule_action
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond/interaction is a hydrogen bond.

        This method checks if the given atom pair could form a hydrogen bond
        based on element types and geometry.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the atoms.

        Returns:
            True if this appears to be a hydrogen bond interaction.
        """
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        # Check if one atom is H and other is an acceptor
        if atom1.symbol == "H":
            h_idx, a_idx = bond[0], bond[1]
            if atom2.symbol not in self.ACCEPTOR_ELEMENTS:
                return False
        elif atom2.symbol == "H":
            h_idx, a_idx = bond[1], bond[0]
            if atom1.symbol not in self.ACCEPTOR_ELEMENTS:
                return False
        else:
            # Neither is H - could be checking D...A distance
            # For now, skip these cases
            return False

        # Find the donor (heavy atom bonded to H)
        donor_idx = None
        for bond_info in system.bonds:
            if bond_info["atom1"] == h_idx:
                donor_idx = bond_info["atom2"]
                break
            elif bond_info["atom2"] == h_idx:
                donor_idx = bond_info["atom1"]
                break

        if donor_idx is None:
            return False

        donor = system.atoms[donor_idx]
        if donor.symbol not in self.DONOR_ELEMENTS:
            return False

        # Check distance (H to acceptor)
        h_coords = system.atoms[h_idx].coords
        a_coords = system.atoms[a_idx].coords

        distance = math.sqrt(sum((h - a) ** 2 for h, a in zip(h_coords, a_coords)))

        if distance > self.distance_cutoff:
            return False

        # Check angle (D-H...A)
        d_coords = donor.coords
        # Calculate angle using vectors
        vec_dh = [h - d for h, d in zip(h_coords, d_coords)]
        vec_ha = [a - h for a, h in zip(a_coords, h_coords)]

        dot = sum(v1 * v2 for v1, v2 in zip(vec_dh, vec_ha))
        mag_dh = math.sqrt(sum(v ** 2 for v in vec_dh))
        mag_ha = math.sqrt(sum(v ** 2 for v in vec_ha))

        if mag_dh == 0 or mag_ha == 0:
            return False

        cos_angle = dot / (mag_dh * mag_ha)
        # Clamp to valid range for acos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        return angle_deg >= self.angle_cutoff

    def action(self) -> RuleAction:
        """Return the configured action for hydrogen bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action
