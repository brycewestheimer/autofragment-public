# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for graph partitioning."""
import networkx as nx
import pytest

from autofragment.algorithms.graph_partition import (
    balanced_partition,
    community_partition,
    hierarchical_decomposition,
    metis_partition,
    min_cut_partition,
)


class MockMolecularGraph:
    """Mock for MolecularGraph that exposes internal graph."""
    def __init__(self, graph):
        self._graph = graph

    def nodes(self):
        return self._graph.nodes()

    def neighbors(self, node):
        return self._graph.neighbors(node)


class MockMolecularGraphNoNodes:
    """Mock exposing only internal graph, like MolecularGraph."""

    def __init__(self, graph):
        self._graph = graph

def test_min_cut_partition_simple():
    """Test standard 2-way partition on a barbell graph."""
    g = nx.Graph()
    # Clique 1
    g.add_edges_from([(0,1), (1,2), (0,2)])
    # Clique 2
    g.add_edges_from([(3,4), (4,5), (3,5)])
    # Bridge
    g.add_edge(2, 3, order=1)

    # Set weights for clique edges higher to ensure bridge is cut
    for u, v in [(0,1), (1,2), (0,2), (3,4), (4,5), (3,5)]:
        g[u][v]['order'] = 10

    mol_graph = MockMolecularGraph(g)

    partitions, cut_edges = min_cut_partition(mol_graph, n_partitions=2, weight_attr='order')

    assert len(partitions) == 2
    # Should be separated into cliques
    p1_set = set(partitions[0])
    assert p1_set == {0, 1, 2} or p1_set == {3, 4, 5}
    assert len(cut_edges) == 1
    assert (2, 3) in cut_edges or (3, 2) in cut_edges

def test_recursive_partition():
    """Test 4-way partition."""
    # 4 cliques connected in a line
    g = nx.Graph()
    cliques = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ]

    # Internal edges
    for c in cliques:
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                g.add_edge(c[i], c[j], order=10)

    # Connecting edges
    g.add_edge(2, 3, order=1)
    g.add_edge(5, 6, order=1)
    g.add_edge(8, 9, order=1)

    mol_graph = MockMolecularGraph(g)

    partitions, cut_edges = min_cut_partition(mol_graph, n_partitions=4, weight_attr='order')

    assert len(partitions) == 4
    assert len(cut_edges) == 3

def test_partition_disconnected():
    """Test partitioning a disconnected graph."""
    g = nx.Graph()
    g.add_edge(0, 1, order=1)
    g.add_edge(2, 3, order=1)

    mol_graph = MockMolecularGraph(g)
    partitions, cut_edges = min_cut_partition(mol_graph, n_partitions=2, weight_attr='order')

    assert len(partitions) == 2
    assert len(cut_edges) == 0

def test_community_label_prop():
    """Test label propagation community detection."""
    # Two cliques connected by one edge
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2), (0,2)])
    g.add_edges_from([(3,4), (4,5), (3,5)])
    g.add_edge(2, 3)

    mol_graph = MockMolecularGraph(g)

    partitions, cut_edges = community_partition(mol_graph, algorithm="label_propagation")

    # Likely finds 2 communities
    assert len(partitions) == 2
    assert (2, 3) in cut_edges or (3, 2) in cut_edges

def test_community_louvain_missing():
    """Test ValueError when louvain missing."""
    g = nx.Graph()
    mol_graph = MockMolecularGraph(g)

    # Since we know it's missing, it should raise ImportError
    with pytest.raises(ImportError, match="python-louvain"):
        community_partition(mol_graph, algorithm="louvain")

def test_community_unknown_algo():
    """Test unknown algorithm raises ValueError."""
    g = nx.Graph()
    mol_graph = MockMolecularGraph(g)

    with pytest.raises(ValueError, match="Unknown algorithm"):
        community_partition(mol_graph, algorithm="magic")

def test_balanced_partition_simple():
    """Test balanced partitioning on a line graph."""
    # 0-1-2-3-4-5
    # Should split into [0,1,2] and [3,4,5]
    g = nx.path_graph(6)
    mol_graph = MockMolecularGraph(g)

    partitions, _ = balanced_partition(mol_graph, n_partitions=2)

    assert len(partitions) == 2
    assert len(partitions[0]) == 3
    assert len(partitions[1]) == 3

def test_balanced_partition_recursive():
    """Test 4-way balanced partition."""
    # 0-1...-11 (12 nodes)
    # Target: 4 partitions of size 3
    g = nx.path_graph(12)
    mol_graph = MockMolecularGraph(g)

    partitions, _ = balanced_partition(mol_graph, n_partitions=4)

    assert len(partitions) == 4
    # KL is heuristic, might not be perfect. Allow +/- 1.
    for p in partitions:
        assert 2 <= len(p) <= 4

def test_balanced_partition_disconnected():
    """Test balancing disconnected components."""
    # Two identical components of size 4
    g = nx.disjoint_union(nx.path_graph(4), nx.path_graph(4))
    mol_graph = MockMolecularGraph(g)

    partitions, _ = balanced_partition(mol_graph, n_partitions=2)

    assert len(partitions) == 2
    # Should group one component in each partition to achieve perfect balance
    assert len(partitions[0]) == 4
    assert len(partitions[1]) == 4

def test_hierarchical_decomposition():
    # 0-1-2-3
    g = nx.path_graph(4)
    mol_graph = MockMolecularGraph(g)

    # Should split into 2 children (0-1, 2-3), each splitting into 2 leafs (size 1)
    # min_size=1 means we go down to single atoms
    # 0-1-2-3 (level 0)
    # |-> 0-1 (level 1)
    #     |-> 0 (level 2)
    #     |-> 1
    # |-> 2-3
    #     |-> 2
    #     |-> 3

    tree = hierarchical_decomposition(mol_graph, min_fragment_size=1, max_levels=5)

    assert len(tree.atoms) == 4
    assert len(tree.children) == 2
    assert tree.level == 0
    assert tree.children[0].level == 1

    # Check leaves
    leaves = [node for node in tree.children if node.is_leaf]
    # Children of root are not leaves in this case, they split further
    assert len(leaves) == 0

    # Check deeper
    assert len(tree.children[0].children) == 2


def test_hierarchical_decomposition_without_nodes_method():
    """Regression: hierarchical decomposition should not require graph.nodes()."""
    g = nx.path_graph(4)
    mol_graph = MockMolecularGraphNoNodes(g)

    tree = hierarchical_decomposition(mol_graph, min_fragment_size=1, max_levels=5)

    assert len(tree.atoms) == 4
    assert len(tree.children) == 2

def test_disconnected_fewer_components_than_partitions():
    """Test: 2 components, request 4 partitions -> largest gets split."""
    g = nx.Graph()
    # Component 1: 6 nodes (path)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    for u, v in g.edges():
        g[u][v]["order"] = 1
    # Component 2: 2 nodes
    g.add_edge(10, 11, order=1)

    mol_graph = MockMolecularGraph(g)
    partitions, _ = min_cut_partition(mol_graph, n_partitions=4, weight_attr="order")

    assert len(partitions) == 4
    # All original nodes should appear exactly once
    all_nodes = sorted(n for p in partitions for n in p)
    assert all_nodes == [0, 1, 2, 3, 4, 5, 10, 11]


def test_disconnected_single_node_components():
    """Test: 3 single-node components, request 4 -> warning, returns 3."""
    import warnings

    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)

    mol_graph = MockMolecularGraph(g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        partitions, _ = min_cut_partition(mol_graph, n_partitions=4, weight_attr="order")
        assert any("Cannot split further" in str(warning.message) for warning in w)
    assert len(partitions) == 3


def test_metis_fallback():
    # Since we likely don't have pymetis, this tests fallback to balanced_partition
    g = nx.path_graph(4)
    mol_graph = MockMolecularGraph(g)

    partitions, _ = metis_partition(mol_graph, n_partitions=2)
    assert len(partitions) == 2

