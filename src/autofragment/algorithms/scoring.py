# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Fragmentation scoring system."""
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from autofragment.rules.base import RuleAction

if TYPE_CHECKING:
    from autofragment.core.types import ChemicalSystem
    from autofragment.rules.base import RuleEngine

@dataclass
class FragmentationScore:
    """Container for all fragmentation quality metrics.

    Total score is computed as weighted sum of component scores.
    Higher total score = better fragmentation.

    Attributes:
        bond_penalty: Penalty for broken bonds (lower is better, negated)
        size_variance: Penalty for unequal fragment sizes (lower is better)
        interface_score: Score for minimizing interface atoms (higher is better)
        integrity_score: Chemical integrity preservation (higher is better)
        cost_estimate: Estimated computational cost (lower is better)

    Example:
        >>> score = FragmentationScore(
        ...     bond_penalty=2.5,
        ...     size_variance=0.1,
        ...     interface_score=0.8,
        ...     integrity_score=0.95
        ... )
        >>> score.total()
        -1.85
    """

    bond_penalty: float = 0.0
    size_variance: float = 0.0
    interface_score: float = 0.0
    integrity_score: float = 0.0
    cost_estimate: float = 0.0

    # Additional metadata
    n_fragments: int = 0
    n_broken_bonds: int = 0
    fragment_sizes: List[int] = field(default_factory=list)

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total score.

        Args:
            weights: Dict mapping component names to weights.
                     Missing components use default weight of 1.0.

        Returns:
            Weighted sum of all components (penalties are subtracted).
        """
        if weights is None:
            weights = {}

        def w(key: str) -> float:
            """Return a normalized weight for a scoring component."""
            return weights.get(key, 1.0)

        return (
            - w("bond_penalty") * self.bond_penalty
            - w("size_variance") * self.size_variance
            + w("interface_score") * self.interface_score
            + w("integrity_score") * self.integrity_score
            - w("cost_estimate") * self.cost_estimate
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return asdict(self)

def bond_breaking_penalty(
    bond: Tuple[int, int],
    system: "ChemicalSystem",
    rule_engine: Optional["RuleEngine"] = None
) -> float:
    """Calculate penalty for breaking a specific bond.

    Penalties are based on:
    - Bond order (higher order = higher penalty)
    - Atom types involved
    - Rule engine constraints (if provided)

    Args:
        bond: Tuple of atom indices
        system: ChemicalSystem containing the bond
        rule_engine: Optional rules to consider

    Returns:
        Penalty value (0.0 = fine to break, higher = worse)
    """
    graph = system.graph
    bond_data = graph.get_bond(*bond)

    if bond_data is None:
        return 0.0

    # Base penalty from bond order
    order = bond_data.get("order", 1.0)
    penalty = order * 1.0  # Single=1, Double=2, Triple=3

    # Aromatic penalty
    # Assume bond_type attribute is 'aromatic' or check if boolean 'is_aromatic' exists?
    # Task spec says: bond_data.get("bond_type") == "aromatic"
    if bond_data.get("bond_type") == "aromatic":
        penalty += 5.0  # Heavy penalty for aromatic

    # Check rule constraints
    if rule_engine:
        # Check evaluate_bond method availability or similar
        action = rule_engine.evaluate_bond(bond, system)
        if action == RuleAction.MUST_NOT_BREAK:
            penalty = float('inf')
        elif action == RuleAction.PREFER_KEEP:
            penalty *= 2.0
        elif action == RuleAction.PREFER_BREAK:
            penalty *= 0.5

    return penalty


def total_bond_penalty(
    broken_bonds: List[Tuple[int, int]],
    system: "ChemicalSystem",
    rule_engine: Optional["RuleEngine"] = None
) -> float:
    """Calculate total penalty for all broken bonds."""
    return sum(
        bond_breaking_penalty(bond, system, rule_engine)
        for bond in broken_bonds
    )

def size_variance_penalty(
    fragment_sizes: List[int],
    target_size: Optional[int] = None
) -> float:
    """Calculate penalty for unequal fragment sizes.

    Uses variance/std to measure size imbalance.

    Args:
        fragment_sizes: List of atom counts per fragment
        target_size: Optional target size (uses mean if not given)

    Returns:
        Variance-based penalty (0.0 = all equal, higher = more variance)
    """
    if len(fragment_sizes) < 2:
        return 0.0

    sizes = np.array(fragment_sizes)
    target = target_size if target_size else sizes.mean()

    # Normalized variance (coefficient of variation)
    if target > 0:
        cv = sizes.std() / target
    else:
        cv = 0.0

    return float(cv)

def interface_score(
    broken_bonds: List[Tuple[int, int]],
    system: "ChemicalSystem"
) -> float:
    """Calculate score for minimizing interface atoms.

    Interface atoms are those involved in broken bonds.
    Fewer interface atoms = better score.

    Args:
        broken_bonds: Bonds cut during fragmentation
        system: Original chemical system

    Returns:
        Score between 0 and 1 (1.0 = minimal interface, higher is better)
    """
    if not broken_bonds:
        return 1.0

    # Count atoms at interfaces (involved in broken bonds)
    interface_atoms: Set[int] = set()
    for i, j in broken_bonds:
        interface_atoms.add(i)
        interface_atoms.add(j)

    total_atoms = system.n_atoms
    if total_atoms == 0:
        return 1.0

    interface_ratio = len(interface_atoms) / total_atoms

    # Invert so higher = better
    return 1.0 - interface_ratio

def chemical_integrity_score(
    partition: List[List[int]],
    system: "ChemicalSystem"
) -> float:
    """Calculate chemical integrity preservation score.

    Measures how well fragments preserve:
    - Complete rings (aromatic, aliphatic)

    Args:
        partition: List of lists of atom indices (the fragmentation)
        system: Original ChemicalSystem

    Returns:
        Score between 0 and 1 (1.0 = perfect integrity)
    """
    # Find all rings in the system
    rings = system.graph.find_rings()
    if not rings:
        return 1.0

    # Create a lookup for atom -> fragment_index
    atom_to_frag = {}
    for i, frag_indices in enumerate(partition):
        for atom_idx in frag_indices:
            atom_to_frag[atom_idx] = i

    # Check each ring
    preserved_rings = 0
    for ring in rings:
        # Get fragment IDs for all atoms in ring
        frag_ids = {atom_to_frag.get(a) for a in ring if a in atom_to_frag}

        # If ring is fully contained in one fragment (len(frag_ids) == 1), it's preserved
        if len(frag_ids) == 1:
            preserved_rings += 1

    return preserved_rings / len(rings)

def computational_cost_estimate(
    fragment_sizes: List[int],
    method: str = "dft",
    scaling: float = 3.0
) -> float:
    """Estimate relative computational cost.

    QC methods scale as O(N^scaling) where N is system size.
    Fragment-based methods trade many small calculations for one large.

    Args:
        fragment_sizes: Atom counts per fragment
        method: Computational method (affects scaling)
        scaling: Power-law scaling exponent (default 3.0 for DFT)

    Returns:
        Relative cost (lower is better)
    """
    if not fragment_sizes:
        return 0.0

    # Cost of full calculation
    total_atoms = sum(fragment_sizes)
    if total_atoms == 0:
        return 0.0

    full_cost = total_atoms ** scaling

    # Cost of fragmented calculation
    fragment_cost = sum(n ** scaling for n in fragment_sizes)

    # Return ratio (fragment cost / full cost)
    # Values < 1.0 mean fragmentation is beneficial
    return fragment_cost / full_cost

@dataclass
class ScoringWeights:
    """Configurable weights for scoring components.

    All weights default to 1.0. Set to 0.0 to ignore a component.
    Higher weight = more importance in total score.

    Example:
        >>> weights = ScoringWeights(
        ...     bond_penalty=2.0,  # Heavily penalize breaking bonds
        ...     size_variance=0.5  # Less important
        ... )
        >>> score.total(weights.to_dict())
    """

    bond_penalty: float = 1.0
    size_variance: float = 1.0
    interface_score: float = 1.0
    integrity_score: float = 1.0
    cost_estimate: float = 1.0

    @classmethod
    def for_fmo(cls) -> "ScoringWeights":
        """Preset weights optimized for FMO calculations."""
        # FMO prefers maintaining integrity and respecting bond penalties
        return cls(
            bond_penalty=2.0,
            size_variance=1.5,
            interface_score=1.0,
            integrity_score=2.0,
            cost_estimate=0.5
        )

    @classmethod
    def for_mbe(cls) -> "ScoringWeights":
        """Preset weights optimized for many-body expansion."""
        # MBE often benefits from equal-sized clusters
        return cls(
            bond_penalty=1.0,
            size_variance=2.0,  # Equal sizes important
            interface_score=0.5,
            integrity_score=1.0,
            cost_estimate=1.5
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for FragmentationScore.total()."""
        return asdict(self)
