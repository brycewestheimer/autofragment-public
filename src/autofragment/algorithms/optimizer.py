# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Fragmentation optimization algorithms."""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from autofragment.algorithms.scoring import ScoringWeights
    from autofragment.core.types import ChemicalSystem
    from autofragment.rules.base import RuleEngine


@dataclass
class FragmentationResult:
    """Result of fragmentation optimization."""

    fragments: List[List[int]]
    broken_bonds: List[Tuple[int, int]]
    score: float


@dataclass
class ConvergenceCriteria:
    """Configurable convergence conditions.

    Optimizer stops when ANY condition is met.
    """

    max_iterations: int = 1000
    max_time_seconds: float = 300.0  # 5 minutes
    min_improvement: float = 1e-6  # Stop if delta < this
    patience: int = 50  # Iterations without improvement
    target_score: Optional[float] = None  # Stop if score reached

    def should_stop(
        self,
        iteration: int,
        elapsed_time: float,
        current_score: float,
        best_score: float,
        iterations_since_improvement: int,
    ) -> Tuple[bool, str]:
        """Check if optimization should stop.

        Returns:
            Tuple of (should_stop, reason)
        """
        if iteration >= self.max_iterations:
            return True, "max_iterations"
        if elapsed_time >= self.max_time_seconds:
            return True, "timeout"
        if iterations_since_improvement >= self.patience:
            return True, "no_improvement"
        if self.target_score and current_score >= self.target_score:
            return True, "target_reached"
        return False, ""


class FragmentationOptimizer(ABC):
    """Base class for fragmentation optimizers."""

    @abstractmethod
    def optimize(
        self,
        system: "ChemicalSystem",
        rule_engine: Optional["RuleEngine"] = None,
        weights: Optional["ScoringWeights"] = None,
    ) -> FragmentationResult:
        """Find optimal fragmentation."""
        pass


class GreedyOptimizer(FragmentationOptimizer):
    """Greedy fragmentation with local search refinement.

    Algorithm:
    1. Start with all atoms in one fragment
    2. Greedily select best bond to break (lowest penalty)
    3. Repeat until target fragment count or no improvement
    4. Apply local search to refine solution
    """

    def __init__(
        self,
        target_fragments: Optional[int] = None,
        max_iterations: int = 1000,
        local_search_steps: int = 100,
    ):
        """Initialize a new GreedyOptimizer instance."""
        self.target_fragments = target_fragments
        self.max_iterations = max_iterations
        self.local_search_steps = local_search_steps

    def optimize(
        self,
        system: "ChemicalSystem",
        rule_engine: Optional["RuleEngine"] = None,
        weights: Optional["ScoringWeights"] = None,
    ) -> FragmentationResult:
        """Execute greedy optimization."""
        # TODO: Implement full scoring integration.
        # For now, simplistic bond penalty based greedy.

        from autofragment.algorithms.scoring import bond_breaking_penalty, total_bond_penalty

        # Initial: one fragment with all atoms
        all_atoms = list(range(system.n_atoms))
        fragments = [all_atoms]
        broken_bonds = []

        # Available bonds to break (that are not yet broken)
        # We need to track bonds within fragments
        potential_bonds = self._get_bonds_in_fragments(system, fragments)

        for _ in range(self.max_iterations):
            # Stopping check
            if self.target_fragments and len(fragments) >= self.target_fragments:
                break

            if not potential_bonds:
                break

            # Select best bond to break
            # Heuristic: break bond with min penalty that yields a split
            # Note: simplistic greedy doesn't look ahead at resulting fragment quality
            # Improved greedy would check score improvement.

            best_bond = None
            min_penalty = float("inf")

            for bond in potential_bonds:
                penalty = bond_breaking_penalty(bond, system, rule_engine)
                if penalty < min_penalty:
                    # Verify this bond actually splits a fragment
                    if self._bond_splits_fragment(bond, fragments, system):
                        min_penalty = penalty
                        best_bond = bond

            if best_bond is None:
                break

            # Apply break
            broken_bonds.append(best_bond)
            fragments = self._update_fragments(fragments, best_bond, system)
            potential_bonds = self._get_bonds_in_fragments(system, fragments)

        # Local search (placeholder for now)
        # fragments = self._local_search(fragments, system, weights)

        # Calculate final score
        # For now just bond penalty
        final_score = total_bond_penalty(broken_bonds, system)

        return FragmentationResult(fragments, broken_bonds, final_score)

    def _get_bonds_in_fragments(
        self, system: "ChemicalSystem", fragments: List[List[int]]
    ) -> List[Tuple[int, int]]:
        """Get all bonds where both atoms are in the same fragment."""
        bonds = []
        # Map atom to fragment index
        atom_to_frag = {}
        for i, frag in enumerate(fragments):
            for atom in frag:
                atom_to_frag[atom] = i

        # Iterate system bonds
        # system.bonds is list of dicts with atom1, atom2
        if hasattr(system, "to_graph"):
            # If we have graph access, use it?
            # Or use raw bonds list
            pass

        for bond_info in system.bonds:
            u, v = bond_info["atom1"], bond_info["atom2"]
            if atom_to_frag[u] == atom_to_frag[v]:
                bonds.append((u, v))

        return bonds

    def _bond_splits_fragment(
        self, bond: Tuple[int, int], fragments: List[List[int]], system
    ) -> bool:
        """Check if breaking a bond splits a fragment into disconnected components."""
        u, v = bond
        # Find which fragment contains u (and v)
        target_frag = None
        for frag in fragments:
            if u in frag:
                target_frag = frag
                break

        if not target_frag:
            return False

        # Build subgraph of this fragment minus the bond
        # Slow for python loop, but functional
        # Using networkx would be better if system.graph available
        # TODO: Optimize with graph
        # For now, optimistic return True if we assume trees or similar?
        # No, cycles exist.

        # Let's use simple reachability (BFS)
        # Build adj list for fragment
        # We need edges within this fragment, excluding (u,v)
        # Scan all system bonds again? Expensive.
        # Assuming we have a graph on system would be O(1) per edge check
        # Let's assume passed system has a .graph attribute (MolecularGraph) as per previous tasks

        if not hasattr(system, "graph"):
            raise ValueError(
                "ChemicalSystem must have a graph attribute to check bond splits. "
                "Ensure the system has bonds or call system.graph to build the graph."
            )

        sg = system.graph.subgraph(set(target_frag))
        if sg.has_edge(u, v):
            sg.remove_edge(u, v)
        if sg.has_edge(v, u):
            sg.remove_edge(v, u)
        return not sg.is_connected()

    def _update_fragments(
        self, fragments: List[List[int]], broken_bond: Tuple[int, int], system
    ) -> List[List[int]]:
        """Splits the fragment containing the bond."""
        u, v = broken_bond
        new_fragments = []
        target_frag_idx = -1

        for i, frag in enumerate(fragments):
            if u in frag and v in frag:
                target_frag_idx = i
                break
            new_fragments.append(frag)

        if target_frag_idx != -1:
            # Split this fragment
            nx = import_nx()

            target_frag = fragments[target_frag_idx]
            sg = system.graph.subgraph(set(target_frag))
            if sg.has_edge(u, v):
                sg.remove_edge(u, v)

            # Get connected components
            ccs = list(nx.connected_components(sg.networkx_graph))
            for cc in ccs:
                new_fragments.append(list(cc))

        return new_fragments


class SimulatedAnnealingOptimizer(FragmentationOptimizer):
    """Simulated annealing for global optimization.

    Algorithm:
    1. Initialize with greedy solution
    2. Each iteration: pick a random atom at a fragment boundary, move it
       to an adjacent fragment
    3. Accept the move if it improves the score, or with Metropolis
       probability exp(-delta / temp)
    4. Cool temperature by cooling_rate each iteration
    5. Track and return the best solution found
    """

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
        max_iterations: int = 10000,
        target_fragments: int = 2,
    ):
        """Initialize a new SimulatedAnnealingOptimizer instance."""
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.target_fragments = target_fragments

    def optimize(
        self,
        system: "ChemicalSystem",
        rule_engine: Optional["RuleEngine"] = None,
        weights: Optional["ScoringWeights"] = None,
    ) -> FragmentationResult:
        """Run simulated annealing optimisation."""
        from autofragment.algorithms.scoring import total_bond_penalty

        nx = import_nx()

        # Seed with greedy solution
        greedy = GreedyOptimizer(target_fragments=self.target_fragments)
        current = greedy.optimize(system, rule_engine, weights)

        current_fragments = [list(f) for f in current.fragments]
        current_score = current.score

        best_fragments = [list(f) for f in current_fragments]
        best_score = current_score
        best_broken = list(current.broken_bonds)

        temp = self.initial_temp

        for _ in range(self.max_iterations):
            if temp < self.min_temp:
                break

            # Attempt a random boundary move
            candidate = self._random_modification(current_fragments, system, nx)
            if candidate is None:
                temp *= self.cooling_rate
                continue

            new_fragments, new_broken = candidate
            new_score = total_bond_penalty(new_broken, system, rule_engine)

            delta = new_score - current_score

            # Metropolis criterion (lower score is better)
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_fragments = new_fragments
                current_score = new_score

                if current_score < best_score:
                    best_fragments = [list(f) for f in current_fragments]
                    best_score = current_score
                    best_broken = list(new_broken)

            temp *= self.cooling_rate

        return FragmentationResult(best_fragments, best_broken, best_score)

    @staticmethod
    def _random_modification(
        fragments: List[List[int]],
        system: "ChemicalSystem",
        nx,
    ) -> Optional[Tuple[List[List[int]], List[Tuple[int, int]]]]:
        """Move a random boundary atom to an adjacent fragment.

        Returns None if no valid move is found.
        """
        if len(fragments) < 2:
            return None

        mg = system.graph

        # Build atom -> fragment index lookup
        atom_to_frag: dict[int, int] = {}
        for idx, frag in enumerate(fragments):
            for a in frag:
                atom_to_frag[a] = idx

        # Find boundary atoms: atoms whose neighbours include atoms in a
        # different fragment.
        boundary: List[Tuple[int, int]] = []  # (atom, neighbour_frag_idx)
        for idx, frag in enumerate(fragments):
            for a in frag:
                for nbr in mg.neighbors(a):
                    nbr_frag = atom_to_frag.get(nbr)
                    if nbr_frag is not None and nbr_frag != idx:
                        boundary.append((a, nbr_frag))

        if not boundary:
            return None

        # Pick a random boundary atom and target fragment
        atom, target_frag_idx = random.choice(boundary)
        source_frag_idx = atom_to_frag[atom]

        # Ensure moving this atom doesn't disconnect its source fragment
        source_frag = fragments[source_frag_idx]
        if len(source_frag) <= 1:
            return None

        remaining = [a for a in source_frag if a != atom]
        sub = mg.subgraph(set(remaining))
        if not sub.is_connected():
            return None

        # Execute the move
        new_fragments = []
        for i, frag in enumerate(fragments):
            if i == source_frag_idx:
                new_fragments.append([a for a in frag if a != atom])
            elif i == target_frag_idx:
                new_fragments.append(frag + [atom])
            else:
                new_fragments.append(list(frag))

        # Compute new broken bonds
        new_atom_to_frag: dict[int, int] = {}
        for i, frag in enumerate(new_fragments):
            for a in frag:
                new_atom_to_frag[a] = i
        new_broken: List[Tuple[int, int]] = []
        for bond in system.bonds:
            u, v = bond["atom1"], bond["atom2"]
            if new_atom_to_frag.get(u) != new_atom_to_frag.get(v):
                new_broken.append((u, v))

        return new_fragments, new_broken


def import_nx():
    """Import `networkx` lazily and return the module."""
    from autofragment.optional import require_dependency

    return require_dependency("networkx", "graph", "Optimization routines")
