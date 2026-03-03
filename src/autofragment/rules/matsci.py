# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Materials science fragmentation rules.

This module provides rules for solid-state and extended materials systems,
including silica, zeolites, metal-organic frameworks (MOFs), and perovskites.

These rules handle the unique bonding patterns found in materials science
applications where periodic structures and inorganic bonding dominate.
"""
import math
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

from .base import FragmentationRule, RuleAction

if TYPE_CHECKING:
    from autofragment.core.types import ChemicalSystem


class SiloxaneBridgeRule(FragmentationRule):
    """Handle Si-O-Si siloxane bridge bonds.

    Common in silica, zeolites, and mesoporous silica nanoparticles (MSN).
    Typically we want to keep SiO4 tetrahedra intact but may need
    to break at bridging oxygens for fragmentation.

    Bridging oxygen: O atom bonded to exactly 2 Si atoms.
    Terminal/silanol oxygen: O atom bonded to 1 Si and 1 H.

    Attributes:
        name: Rule identifier ("siloxane_bridge")

    Example:
        >>> rule = SiloxaneBridgeRule(rule_action=RuleAction.PREFER_BREAK)
        >>> rule.applies_to(si_o_bond, silica_system)
        True
    """

    name = "siloxane_bridge"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.ALLOW,
        priority: Optional[int] = None
    ):
        """Initialize the siloxane bridge rule.

        Args:
            rule_action: Action to apply to bridging Si-O bonds.
                        Default is ALLOW (neutral).
            priority: Rule priority. Defaults to PRIORITY_MEDIUM.
        """
        super().__init__(priority=priority or self.PRIORITY_MEDIUM)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is Si-O in a bridging context.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if this is a bridging Si-O bond.
        """
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])

        elements = {atom1["element"], atom2["element"]}
        if elements != {"Si", "O"}:
            return False

        # Identify oxygen atom
        if atom1["element"] == "O":
            o_idx = bond[0]
        else:
            o_idx = bond[1]

        return self._is_bridging_oxygen(o_idx, system)

    def _is_bridging_oxygen(self, o_idx: int, system: "ChemicalSystem") -> bool:
        """Check if oxygen is bridging (connected to 2 Si atoms).

        Args:
            o_idx: Index of the oxygen atom.
            system: ChemicalSystem containing the atom.

        Returns:
            True if oxygen is bonded to exactly 2 Si atoms.
        """
        si_neighbors = 0
        for bond_info in system.bonds:
            if bond_info["atom1"] == o_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == o_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if system.atoms[neighbor_idx].symbol == "Si":
                si_neighbors += 1

        return si_neighbors == 2

    def action(self) -> RuleAction:
        """Return the configured action for siloxane bridges.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class MOFLinkerRule(FragmentationRule):
    """Preserve organic linker molecules in MOFs.

    MOFs (Metal-Organic Frameworks) consist of metal nodes connected
    by organic linkers. Common linkers include:
    - BDC (1,4-benzenedicarboxylate)
    - BTC (1,3,5-benzenetricarboxylate)
    - Imidazolate (for ZIF structures)

    Bonds within organic linkers should remain intact during fragmentation,
    while metal-linker bonds may be break points.

    Attributes:
        name: Rule identifier ("mof_linker")
        organic_elements: Set of elements considered organic

    Example:
        >>> rule = MOFLinkerRule()
        >>> rule.applies_to(c_c_bond_in_bdc, mof_system)
        True
    """

    name = "mof_linker"

    # Elements that make up organic linkers (excluding metals)
    ORGANIC_ELEMENTS: Set[str] = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}

    # Common metals in MOFs
    MOF_METALS: Set[str] = {
        "Zn", "Cu", "Fe", "Co", "Ni", "Mn", "Cr", "Al", "Mg",
        "Ti", "Zr", "Hf", "V", "Mo", "W", "Cd", "Pb"
    }

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        organic_elements: Optional[Set[str]] = None,
        priority: Optional[int] = None
    ):
        """Initialize the MOF linker rule.

        Args:
            rule_action: Action to apply to linker bonds.
                        Default is MUST_NOT_BREAK.
            organic_elements: Set of elements considered organic.
                            Uses ORGANIC_ELEMENTS if None.
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action
        self.organic_elements = organic_elements or self.ORGANIC_ELEMENTS.copy()

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is within an organic linker.

        A linker bond is one where both atoms are organic elements
        (not metals).

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if both atoms are organic (non-metal) elements.
        """
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        # Both atoms must be organic (non-metal)
        elem1 = atom1.symbol
        elem2 = atom2.symbol

        # If either is a metal, this is NOT a linker bond
        if elem1 in self.MOF_METALS or elem2 in self.MOF_METALS:
            return False

        # Both must be organic elements
        return elem1 in self.organic_elements and elem2 in self.organic_elements

    def action(self) -> RuleAction:
        """Return the configured action for MOF linker bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class MetalNodeRule(FragmentationRule):
    """Preserve metal node coordination in materials.

    Metal nodes in MOFs (Zn4O clusters, Cu paddlewheels, etc.) should
    remain intact during fragmentation. This rule keeps all bonds
    within a metal cluster together.

    Different from MetalCoordinationRule (2.2.3) which handles individual
    metal-ligand bonds. This rule specifically handles multi-metal
    clusters where metals are bridged by oxygen or other atoms.

    Attributes:
        name: Rule identifier ("metal_node")
        cluster_distance: Maximum M-M distance to be considered same cluster

    Example:
        >>> rule = MetalNodeRule()
        >>> rule.applies_to(zn_o_bond, zn4o_cluster)
        True
    """

    name = "metal_node"

    # Common metals in MOF nodes
    NODE_METALS: Set[str] = {
        "Zn", "Cu", "Fe", "Co", "Ni", "Mn", "Cr", "Al",
        "Ti", "Zr", "Hf", "V", "Mo", "W", "Cd"
    }

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        cluster_distance: float = 4.0,  # Max M-M distance in cluster (Å)
        metals: Optional[Set[str]] = None,
        priority: Optional[int] = None
    ):
        """Initialize the metal node rule.

        Args:
            rule_action: Action to apply to bonds within metal nodes.
                        Default is MUST_NOT_BREAK.
            cluster_distance: Maximum metal-metal distance to be considered
                            part of the same cluster (Angstroms).
            metals: Set of metal element symbols. Uses NODE_METALS if None.
            priority: Rule priority. Defaults to PRIORITY_CRITICAL.
        """
        super().__init__(priority=priority or self.PRIORITY_CRITICAL)
        self._action = rule_action
        self.cluster_distance = cluster_distance
        self.metals = metals if metals is not None else self.NODE_METALS.copy()
        # Cache for metal clusters
        self._cluster_cache: Dict[int, Set[int]] = {}

    def _find_metal_clusters(self, system: "ChemicalSystem") -> list[Set[int]]:
        """Find all metal clusters in the system.

        Metals within cluster_distance of each other are grouped together.

        Returns:
            List of sets, each containing atom indices of a metal cluster.
        """
        # Find all metal atoms
        metal_indices = []
        for i, atom in enumerate(system.atoms):
            if atom.symbol in self.metals:
                metal_indices.append(i)

        if not metal_indices:
            return []

        # Group metals by distance (simple union-find approach)
        # Build adjacency based on distance
        adj: Dict[int, Set[int]] = {i: set() for i in metal_indices}

        for i, m1_idx in enumerate(metal_indices):
            for m2_idx in metal_indices[i+1:]:
                m1 = system.atoms[m1_idx]
                m2 = system.atoms[m2_idx]
                dist = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(m1.coords, m2.coords)))
                if dist <= self.cluster_distance:
                    adj[m1_idx].add(m2_idx)
                    adj[m2_idx].add(m1_idx)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        for start in metal_indices:
            if start in visited:
                continue

            cluster = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(cluster) > 1:  # Only count as cluster if >1 metal
                clusters.append(cluster)

        return clusters

    def _get_cluster_atoms(self, system: "ChemicalSystem") -> Set[int]:
        """Get all atoms that are part of metal clusters (including bridging atoms).

        Returns:
            Set of atom indices that are within or bridging metal clusters.
        """
        system_id = id(system)
        if system_id in self._cluster_cache:
            return self._cluster_cache[system_id]

        clusters = self._find_metal_clusters(system)
        cluster_atoms: Set[int] = set()

        for cluster in clusters:
            # Add all metals in cluster
            cluster_atoms.update(cluster)

            # Add atoms bonded to metals in this cluster
            for metal_idx in cluster:
                for bond_info in system.bonds:
                    if bond_info["atom1"] == metal_idx:
                        neighbor = bond_info["atom2"]
                    elif bond_info["atom2"] == metal_idx:
                        neighbor = bond_info["atom1"]
                    else:
                        continue

                    # Include bridging atoms (typically O) that connect
                    # multiple metals in the cluster
                    if system.atoms[neighbor].symbol in {"O", "S", "N"}:
                        # Check if this atom bridges to another metal in cluster
                        for other_bond in system.bonds:
                            if other_bond["atom1"] == neighbor:
                                other_end = other_bond["atom2"]
                            elif other_bond["atom2"] == neighbor:
                                other_end = other_bond["atom1"]
                            else:
                                continue

                            if other_end in cluster and other_end != metal_idx:
                                cluster_atoms.add(neighbor)
                                break

        self._cluster_cache[system_id] = cluster_atoms
        return cluster_atoms

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is within a metal cluster.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            True if both atoms are part of the same metal cluster.
        """
        cluster_atoms = self._get_cluster_atoms(system)
        return bond[0] in cluster_atoms and bond[1] in cluster_atoms

    def action(self) -> RuleAction:
        """Return the configured action for metal node bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class PerovskiteOctahedralRule(FragmentationRule):
    """Preserve MO₆ octahedra in perovskites.

    Perovskites have the general formula ABO₃, with corner-sharing
    BO₆ octahedra as the fundamental structural unit. This rule
    keeps the B-O bonds within octahedra intact.

    Common B-site metals include Ti, Zr, Nb, Ta, Fe, Mn, Co.
    Common A-site cations (Ba, Ca, Sr, Pb, K, Na) are not protected
    by this rule.

    Attributes:
        name: Rule identifier ("perovskite_octahedral")
        b_site_metals: Set of B-site metal elements

    Example:
        >>> rule = PerovskiteOctahedralRule()
        >>> rule.applies_to(ti_o_bond, batio3_system)
        True  # if Ti has octahedral coordination
    """

    name = "perovskite_octahedral"

    # Common B-site metals in perovskites
    B_SITE_METALS: Set[str] = {
        "Ti", "Zr", "Nb", "Ta", "Fe", "Mn", "Co", "Ni",
        "V", "Cr", "Mo", "W", "Ru", "Ir", "Sn", "Pb"
    }

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        b_site_metals: Optional[Set[str]] = None,
        octahedral_tolerance: int = 1,  # Allow 5-7 O neighbors
        allow_corner_breaking: bool = False,
        priority: Optional[int] = None
    ):
        """Initialize the perovskite octahedral rule.

        Args:
            rule_action: Action to apply to B-O bonds in octahedra.
                        Default is MUST_NOT_BREAK.
            b_site_metals: Set of B-site metal elements.
                          Uses B_SITE_METALS if None.
            octahedral_tolerance: Allowed deviation from 6 O neighbors
                                 (e.g., 1 means 5-7 is acceptable).
            allow_corner_breaking: If True, allows breaking B-O bonds that bridge
                                 to another B-site metal.
            priority: Rule priority. Defaults to PRIORITY_HIGH.
        """
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action
        self.b_site_metals = b_site_metals if b_site_metals is not None else self.B_SITE_METALS.copy()
        self.octahedral_tolerance = octahedral_tolerance
        self.allow_corner_breaking = allow_corner_breaking

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is B-O in an octahedral environment."""
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        # Identify B-site metal and oxygen
        if atom1.symbol in self.b_site_metals and atom2.symbol == "O":
            metal_idx = bond[0]
            o_idx = bond[1]
        elif atom2.symbol in self.b_site_metals and atom1.symbol == "O":
            metal_idx = bond[1]
            o_idx = bond[0]
        else:
            return False

        # Check proper octahedral coordination
        if not self._is_octahedral(metal_idx, system):
            return False

        # If we allow breaking corners, we must check if this O bridges two B-sites
        if self.allow_corner_breaking:
            # Check neighbors of O
            b_neighbors = 0
            for bond_info in system.bonds:
                if bond_info["atom1"] == o_idx:
                    n = bond_info["atom2"]
                elif bond_info["atom2"] == o_idx:
                    n = bond_info["atom1"]
                else:
                    continue
                if system.atoms[n].symbol in self.b_site_metals:
                    b_neighbors += 1

            # If O bridges two B-sites (Corner Sharing), and we allow breaking,
            # then this rule DOES NOT APPLY (it returns False, so it doesn't force MUST_NOT_BREAK).
            # This allows other rules (or default ALLOW) to govern it.
            if b_neighbors >= 2:
                return False

        return True

    def _is_octahedral(self, metal_idx: int, system: "ChemicalSystem") -> bool:
        """Check if metal has approximately octahedral O coordination.

        Args:
            metal_idx: Index of the metal atom.
            system: ChemicalSystem containing the atom.

        Returns:
            True if metal has 5-7 oxygen neighbors (octahedral within tolerance).
        """
        o_neighbors = 0
        for bond_info in system.bonds:
            if bond_info["atom1"] == metal_idx:
                neighbor_idx = bond_info["atom2"]
            elif bond_info["atom2"] == metal_idx:
                neighbor_idx = bond_info["atom1"]
            else:
                continue

            if system.atoms[neighbor_idx].symbol == "O":
                o_neighbors += 1

        # Octahedral = 6 O neighbors, with some tolerance
        min_o = 6 - self.octahedral_tolerance
        max_o = 6 + self.octahedral_tolerance
        return min_o <= o_neighbors <= max_o

    def action(self) -> RuleAction:
        """Return the configured action for octahedral bonds.

        Returns:
            The RuleAction specified at initialization.
        """
        return self._action


class ZeoliteAcidSiteRule(FragmentationRule):
    """Preserve Brønsted acid sites (Al-O-H-Si) in zeolites.

    Identifies bridging oxygens between Al and Si that are protonated.
    Both Al-O and Si-O bonds involving the protonated oxygen are protected.

    Attributes:
        name: Rule identifier ("zeolite_acid_site")
    """

    name = "zeolite_acid_site"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        priority: Optional[int] = None
    ):
        """Initialize a new ZeoliteAcidSiteRule instance."""
        super().__init__(priority=priority or self.PRIORITY_CRITICAL)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is part of Al-O(H)-Si site."""
        graph = system.to_graph()
        atom1 = graph.get_atom(bond[0])
        atom2 = graph.get_atom(bond[1])

        # Check elements: must be Al-O or Si-O
        syms = {atom1["element"], atom2["element"]}
        if not (syms == {"Al", "O"} or syms == {"Si", "O"}):
            return False

        # Identify Oxygen
        o_idx = bond[0] if atom1["element"] == "O" else bond[1]

        # Check if Oxygen is protonated (has H neighbor)
        is_protonated = False
        neighbors = []
        for bond_info in system.bonds:
            if bond_info["atom1"] == o_idx:
                n = bond_info["atom2"]
            elif bond_info["atom2"] == o_idx:
                n = bond_info["atom1"]
            else:
                continue
            neighbors.append(n)
            if system.atoms[n].symbol == "H":
                is_protonated = True

        if not is_protonated:
            return False

        # Check if bonded to both Al and Si (among neighbors)
        has_al = False
        has_si = False
        for n in neighbors:
            sym = system.atoms[n].symbol
            if sym == "Al":
                has_al = True
            if sym == "Si":
                has_si = True

        return has_al and has_si

    def action(self) -> RuleAction:
        """Return the rule action for the provided bond."""
        return self._action


class MetalCarboxylateRule(FragmentationRule):
    """Allow breaking of Metal-Oxygen bonds in carboxylate linkers.

    This facilitates separation of metal nodes from organic linkers in MOFs.
    Specifically targets M-O bonds where O is part of a carboxylate group (-COO).

    Attributes:
        name: Rule identifier ("metal_carboxylate")
    """

    name = "metal_carboxylate"

    MOF_METALS: Set[str] = {
        "Zn", "Cu", "Fe", "Co", "Ni", "Mn", "Cr", "Al", "Mg",
        "Ti", "Zr", "Hf", "V", "Mo", "W", "Cd", "Pb"
    }

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.PREFER_BREAK,
        priority: Optional[int] = None
    ):
        # High priority to override default metal coordination rules if needed
        """Initialize a new MetalCarboxylateRule instance."""
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check for M-O-C=O pattern."""
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        # Must be M-O
        syms = {atom1.symbol, atom2.symbol}
        if "O" not in syms:
            return False

        metal_idx = -1
        o_idx = -1

        if atom1.symbol in self.MOF_METALS and atom2.symbol == "O":
            metal_idx = bond[0]
            o_idx = bond[1]
        elif atom2.symbol in self.MOF_METALS and atom1.symbol == "O":
            metal_idx = bond[1]
            o_idx = bond[0]
        else:
            return False

        # Check if O is part of Carboxylate (O-C=O)
        # O bonded to C. That C bonded to another O (double?) and something else.
        # Check O neighbors for C
        c_neighbor = None
        for bond_info in system.bonds:
            if bond_info["atom1"] == o_idx:
                n = bond_info["atom2"]
            elif bond_info["atom2"] == o_idx:
                n = bond_info["atom1"]
            else:
                continue

            if n != metal_idx and system.atoms[n].symbol == "C":
                c_neighbor = n
                break

        if c_neighbor is None:
            return False

        # Check if C is part of carboxylate (bonded to another O)
        is_carboxylate = False
        for bond_info in system.bonds:
            if bond_info["atom1"] == c_neighbor:
                n = bond_info["atom2"]
            elif bond_info["atom2"] == c_neighbor:
                n = bond_info["atom1"]
            else:
                continue

            if n != o_idx and system.atoms[n].symbol == "O":
                # Found the other Oxygen
                is_carboxylate = True
                break

        return is_carboxylate

    def action(self) -> RuleAction:
        """Return the rule action for the provided bond."""
        return self._action


class PolymerBackboneRule(FragmentationRule):
    """Manage polymer backbone fragmentation.

    Identifies bonds within the polymer backbone.
    Can be configured to protect them (within monomer) or break them (between monomers).
    For generic use, we assume backbone C-C bonds are breakable to allow
    monomer separation.

    Attributes:
        name: Rule identifier ("polymer_backbone")
    """

    name = "polymer_backbone"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.ALLOW,
        priority: Optional[int] = None
    ):
        """Initialize a new PolymerBackboneRule instance."""
        super().__init__(priority=priority or self.PRIORITY_MEDIUM)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Simple heuristic: C-C bonds with 2+ Carbon neighbors each."""
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        if atom1.symbol != "C" or atom2.symbol != "C":
            return False

        # Check connectivity (backbone carbons usually have 2 C neighbors)
        def count_c_neighbors(atom_idx):
            """Return or compute count c neighbors."""
            c_count = 0
            for bond_info in system.bonds:
                if bond_info["atom1"] == atom_idx:
                    n = bond_info["atom2"]
                elif bond_info["atom2"] == atom_idx:
                    n = bond_info["atom1"]
                else:
                    continue
                if system.atoms[n].symbol == "C":
                    c_count += 1
            return c_count

        c1_neighbors = count_c_neighbors(bond[0])
        c2_neighbors = count_c_neighbors(bond[1])

        # This is a very rough heuristic for linear chains (PE, etc)
        # Chain internal C has 2 C neighbors. Terminals have 1.
        # Branch points have >2.
        return c1_neighbors >= 2 and c2_neighbors >= 2

    def action(self) -> RuleAction:
        """Return the rule action for the provided bond."""
        return self._action


class SilanolRule(FragmentationRule):
    """Preserve terminal silanol (Si-OH) groups.

    Identifies Si-O bonds where the oxygen is also bonded to a hydrogen.
    These should not be broken, as they represent surface termination
    or defects that define the chemical character of the material.

    Attributes:
        name: Rule identifier ("silanol")
    """

    name = "silanol"

    def __init__(
        self,
        rule_action: RuleAction = RuleAction.MUST_NOT_BREAK,
        priority: Optional[int] = None
    ):
        """Initialize a new SilanolRule instance."""
        super().__init__(priority=priority or self.PRIORITY_HIGH)
        self._action = rule_action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if bond is Si-O(H)."""
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        # Must be Si-O bond
        elems = {atom1.symbol, atom2.symbol}
        if elems != {"Si", "O"}:
            return False

        # Identify oxygen
        if atom1.symbol == "O":
            o_idx = bond[0]
        else:
            o_idx = bond[1]

        # Check if Oxygen is bonded to Hydrogen
        for bond_info in system.bonds:
            if bond_info["atom1"] == o_idx:
                neighbor = bond_info["atom2"]
            elif bond_info["atom2"] == o_idx:
                neighbor = bond_info["atom1"]
            else:
                continue

            if system.atoms[neighbor].symbol == "H":
                return True

        return False

    def action(self) -> RuleAction:
        """Return the rule action for the provided bond."""
        return self._action
