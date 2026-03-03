# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Base classes for fragmentation rules."""
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from autofragment.core.types import ChemicalSystem

class RuleAction(Enum):
    """Actions a rule can specify for a bond.

    Actions are ordered by restrictiveness:
    - MUST_NOT_BREAK: Bond must never be broken (highest priority)
    - PREFER_KEEP: Prefer keeping, but can break if necessary
    - ALLOW: No preference (default, neutral)
    - PREFER_BREAK: Good fragmentation point
    """
    MUST_NOT_BREAK = auto()  # Most restrictive
    PREFER_KEEP = auto()
    ALLOW = auto()           # Neutral
    PREFER_BREAK = auto()    # Least restrictive


class FragmentationRule(ABC):
    """Abstract base class for fragmentation rules.

    All rules must implement:
    - name: Unique identifier for the rule
    - applies_to(): Check if rule applies to a given bond
    - action(): Return the RuleAction when rule applies

    Example subclass:
        >>> class MyRule(FragmentationRule):
        ...     name = "my_rule"
        ...     def applies_to(self, bond, system):
        ...         return some_condition(bond, system)
        ...     def action(self):
        ...         return RuleAction.PREFER_KEEP
    """

    name: str  # Class attribute, unique rule identifier

    # Priority constants
    PRIORITY_CRITICAL = 1000
    PRIORITY_HIGH = 800
    PRIORITY_MEDIUM = 500
    PRIORITY_LOW = 200
    PRIORITY_DEFAULT = 500

    def __init__(self, priority: Optional[int] = None):
        """Initialize a new FragmentationRule instance."""
        self._priority = priority if priority is not None else self.PRIORITY_DEFAULT

    @property
    def priority(self) -> int:
        """Rule priority (higher = evaluated first)."""
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        """Allow runtime priority adjustment."""
        self._priority = value

    @abstractmethod
    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Check if this rule applies to the given bond.

        Args:
            bond: Tuple of atom indices (i, j)
            system: ChemicalSystem containing the bond

        Returns:
            True if rule applies, False otherwise
        """
        pass

    @abstractmethod
    def action(self) -> RuleAction:
        """Return the action to apply when this rule matches.

        Returns:
            RuleAction enum value
        """
        pass

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"{self.__class__.__name__}(priority={self.priority})"


class RuleEngine:
    """Central engine for applying fragmentation rules.

    Manages a collection of rules and evaluates them against bonds
    in a molecular system to determine fragmentation behavior.

    The engine maintains rules in priority order (highest first) and
    uses conflict resolution when multiple rules apply to the same bond.
    The most restrictive action always wins.

    Attributes:
        ACTION_ORDER: Mapping of RuleAction to restrictiveness level
                      (lower number = more restrictive)

    Example:
        >>> from autofragment.rules import RuleEngine, AromaticRingRule, DoubleBondRule
        >>> engine = RuleEngine([AromaticRingRule(), DoubleBondRule()])
        >>> action = engine.evaluate_bond((0, 1), system)
        >>> if action != RuleAction.MUST_NOT_BREAK:
        ...     # Bond can potentially be broken
        ...     pass
    """

    # Action restrictiveness order (most to least restrictive)
    ACTION_ORDER: dict[RuleAction, int] = {
        RuleAction.MUST_NOT_BREAK: 0,  # Most restrictive
        RuleAction.PREFER_KEEP: 1,
        RuleAction.ALLOW: 2,
        RuleAction.PREFER_BREAK: 3,    # Least restrictive
    }

    def __init__(self, rules: Optional[list["FragmentationRule"]] = None):
        """Initialize the rule engine.

        Args:
            rules: Optional list of rules to add. They will be sorted by priority.
        """
        self._rules: list["FragmentationRule"] = []
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: "FragmentationRule") -> None:
        """Add a rule to the engine, maintaining priority order.

        Args:
            rule: The fragmentation rule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)  # Higher priority first

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Args:
            rule_name: The name of the rule to remove.

        Returns:
            True if a rule was removed, False if no rule with that name found.
        """
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < original_len

    def get_rules(self) -> list["FragmentationRule"]:
        """Get all rules in priority order (highest first).

        Returns:
            List of rules sorted by priority.
        """
        return list(self._rules)

    def clear_rules(self) -> None:
        """Remove all rules from the engine."""
        self._rules.clear()

    def resolve_conflict(
        self,
        actions: list[tuple["FragmentationRule", RuleAction]]
    ) -> RuleAction:
        """Resolve conflicting rule actions.

        Strategy: Most restrictive action wins.
        Ties broken by priority (higher priority rule wins).

        Args:
            actions: List of (rule, action) tuples from applicable rules.

        Returns:
            The resolved action to apply.
        """
        if not actions:
            return RuleAction.ALLOW  # Default if no rules apply

        # Sort by restrictiveness, then by priority (for tie-breaking)
        sorted_actions = sorted(
            actions,
            key=lambda x: (self.ACTION_ORDER[x[1]], -x[0].priority)
        )
        return sorted_actions[0][1]

    def evaluate_bond(
        self,
        bond: Tuple[int, int],
        system: "ChemicalSystem"
    ) -> RuleAction:
        """Evaluate all rules for a given bond.

        Applies all rules that match the bond and returns the most
        restrictive action using conflict resolution.

        Args:
            bond: Tuple of atom indices (i, j).
            system: ChemicalSystem containing the bond.

        Returns:
            The most restrictive RuleAction from applicable rules,
            or RuleAction.ALLOW if no rules apply.
        """
        applicable_actions: list[tuple["FragmentationRule", RuleAction]] = []

        for rule in self._rules:
            if rule.applies_to(bond, system):
                applicable_actions.append((rule, rule.action()))

        return self.resolve_conflict(applicable_actions)

    def get_bond_actions(
        self,
        system: "ChemicalSystem"
    ) -> dict[Tuple[int, int], RuleAction]:
        """Evaluate all bonds in a system.

        Args:
            system: ChemicalSystem to evaluate.

        Returns:
            Dictionary mapping bond tuples to their resolved actions.
        """
        results: dict[Tuple[int, int], RuleAction] = {}
        for bond_info in system.bonds:
            bond = (bond_info["atom1"], bond_info["atom2"])
            results[bond] = self.evaluate_bond(bond, system)
        return results

    def get_breakable_bonds(
        self,
        system: "ChemicalSystem"
    ) -> list[Tuple[int, int]]:
        """Get all bonds that CAN be broken (not MUST_NOT_BREAK).

        Args:
            system: ChemicalSystem to evaluate.

        Returns:
            List of bond tuples that are allowed to be broken.
        """
        breakable = []
        for bond_info in system.bonds:
            bond = (bond_info["atom1"], bond_info["atom2"])
            action = self.evaluate_bond(bond, system)
            if action != RuleAction.MUST_NOT_BREAK:
                breakable.append(bond)
        return breakable

    def get_preferred_breaks(
        self,
        system: "ChemicalSystem"
    ) -> list[Tuple[int, int]]:
        """Get bonds with PREFER_BREAK action.

        These are the ideal fragmentation points identified by rules.

        Args:
            system: ChemicalSystem to evaluate.

        Returns:
            List of bond tuples marked as preferred break points.
        """
        preferred = []
        for bond_info in system.bonds:
            bond = (bond_info["atom1"], bond_info["atom2"])
            action = self.evaluate_bond(bond, system)
            if action == RuleAction.PREFER_BREAK:
                preferred.append(bond)
        return preferred

    def __len__(self) -> int:
        """Return the number of rules in the engine."""
        return len(self._rules)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        rule_names = [r.name for r in self._rules[:3]]
        if len(self._rules) > 3:
            rule_names.append(f"...({len(self._rules) - 3} more)")
        return f"RuleEngine(rules=[{', '.join(rule_names)}])"


class RuleSet:
    """A collection of fragmentation rules.

    This class serves as a container for rules that can be shared,
    combined, and passed to a RuleEngine.

    Example:
        >>> rules = RuleSet("my_rules")
        >>> rules.add(AromaticRingRule())
        >>> engine = RuleEngine(rules.rules)
    """

    def __init__(self, name: str, rules: Optional[list[FragmentationRule]] = None):
        """Initialize a new RuleSet instance."""
        self.name = name
        self.rules = rules or []

    def add(self, rule: FragmentationRule) -> None:
        """Add a rule to the set."""
        self.rules.append(rule)

    @classmethod
    def from_rules(cls, rules: list[FragmentationRule], name: str = "custom") -> "RuleSet":
        """Create a RuleSet from a list of rules."""
        return cls(name=name, rules=list(rules))


class BondRule(FragmentationRule):
    """Rule based on specific atom types in a bond.

    Example:
        >>> rule = BondRule("CC", atom1_elem="C", atom2_elem="C", action=RuleAction.PREFER_BREAK)
    """

    def __init__(
        self,
        name: str,
        atom1_elem: str,
        atom2_elem: str,
        action: RuleAction = RuleAction.MUST_NOT_BREAK,
        priority: Optional[int] = None
    ):
        """Initialize a new BondRule instance."""
        super().__init__(priority=priority)
        self.name = name
        self.atom1_elem = atom1_elem
        self.atom2_elem = atom2_elem
        self._action = action

    def applies_to(self, bond: Tuple[int, int], system: "ChemicalSystem") -> bool:
        """Return whether this rule applies to the provided bond."""
        atom1 = system.atoms[bond[0]]
        atom2 = system.atoms[bond[1]]

        elems = {atom1.symbol, atom2.symbol}
        target = {self.atom1_elem, self.atom2_elem}
        return elems == target

    def action(self) -> RuleAction:
        """Return the rule action for the provided bond."""
        return self._action

