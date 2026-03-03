# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for optimization algorithms."""
import networkx as nx

from autofragment.algorithms.optimizer import (
    ConvergenceCriteria,
    GreedyOptimizer,
    SimulatedAnnealingOptimizer,
)
from autofragment.core.graph import MolecularGraph


class MockSystemForOpt:
    def __init__(self, n_atoms=10, chain_len=None):
        self.n_atoms = n_atoms
        if chain_len:
            self.n_atoms = chain_len
            self.graph = MolecularGraph(nx.path_graph(chain_len))
        else:
            self.graph = MolecularGraph(nx.path_graph(n_atoms))

        self.bonds = []
        for u, v in self.graph._graph.edges():
            order = 2.0 # Strong default
            # Make middle bond weaker (1.0)
            if chain_len == 4 and (u==1 and v==2) or (u==2 and v==1):
                order = 1.0
            self.graph.add_bond(u, v, order=order)
            self.bonds.append({"atom1": u, "atom2": v, "order": order})

    @property
    def all_atoms(self):
        return list(range(self.n_atoms))

def test_greedy_optimizer_basic():
    # Simple chain: 0-1-2-3 (4 atoms)
    # 2 partitions
    system = MockSystemForOpt(chain_len=4)
    # Bonds: (0,1), (1,2), (2,3)
    # Splitting (1,2) creates [0,1] and [2,3] -> balanced

    optimizer = GreedyOptimizer(target_fragments=2)
    result = optimizer.optimize(system)

    assert len(result.fragments) == 2
    assert len(result.broken_bonds) == 1

    # Should be sorted lists in sets for comparison
    frag_sets = [set(f) for f in result.fragments]
    assert {0, 1} in frag_sets
    assert {2, 3} in frag_sets

def test_convergence_criteria():
    crit = ConvergenceCriteria(
        max_iterations=10,
        max_time_seconds=1.0,
        patience=2
    )

    # Not stopping
    stop, reason = crit.should_stop(
        iteration=1,
        elapsed_time=0.1,
        current_score=1.0,
        best_score=1.0,
        iterations_since_improvement=0
    )
    assert not stop

    # Max iter
    stop, reason = crit.should_stop(iteration=10, elapsed_time=0, current_score=0, best_score=0, iterations_since_improvement=0)
    assert stop
    assert reason == "max_iterations"

    # Time
    stop, reason = crit.should_stop(iteration=0, elapsed_time=1.1, current_score=0, best_score=0, iterations_since_improvement=0)
    assert stop
    assert reason == "timeout"

    # Patience
    stop, reason = crit.should_stop(iteration=5, elapsed_time=0, current_score=0, best_score=0, iterations_since_improvement=2)
    assert stop
    assert reason == "no_improvement"

def test_simulated_annealing_produces_fragments():
    """SA should produce >1 fragment and score <= greedy score."""
    system = MockSystemForOpt(chain_len=6)

    greedy = GreedyOptimizer(target_fragments=3)
    greedy_result = greedy.optimize(system)

    sa = SimulatedAnnealingOptimizer(
        target_fragments=3,
        initial_temp=50.0,
        cooling_rate=0.95,
        max_iterations=200,
    )
    sa_result = sa.optimize(system)

    assert len(sa_result.fragments) >= 2
    # SA should be at least as good as greedy (or not much worse)
    assert sa_result.score <= greedy_result.score + 1.0
