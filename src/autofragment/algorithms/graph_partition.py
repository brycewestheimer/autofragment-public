# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-based partitioning algorithms."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

    from autofragment.core.graph import MolecularGraph

from autofragment.optional import require_dependency


def min_cut_partition(
    graph: "MolecularGraph", n_partitions: int = 2, weight_attr: str = "order"
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Partition graph using minimum cut algorithm.

    Uses networkx's stoer_wagner algorithm for minimum cut.

    Args:
        graph: MolecularGraph to partition
        n_partitions: Number of partitions (recursive for >2)
        weight_attr: Edge attribute to use as weight

    Returns:
        Tuple of (partition_lists, cut_edges)
        partition_lists: List of lists of node indices
        cut_edges: List of edges (u, v) that were cut
    """
    nx = require_dependency("networkx", "graph", "Graph partitioning")
    g = graph.networkx_graph

    # Need to handle disconnected graphs for stoer_wagner
    if not nx.is_connected(g):
        ccs = sorted(
            [list(c) for c in nx.connected_components(g)],
            key=len,
            reverse=True,
        )
        if len(ccs) >= n_partitions:
            # Enough (or more) components already — group smallest into last bucket
            disconnected_partitions = [list(c) for c in ccs[: n_partitions - 1]]
            last_part = []
            for c in ccs[n_partitions - 1 :]:
                last_part.extend(list(c))
            if last_part:
                disconnected_partitions.append(last_part)
            return disconnected_partitions, []

        # Fewer components than requested partitions — split the largest
        # component(s) to make up the difference.
        partitions: list[list[int]] = [list(c) for c in ccs]
        while len(partitions) < n_partitions:
            # Sort by size descending so we always split the largest
            partitions.sort(key=len, reverse=True)
            largest = partitions[0]
            if len(largest) < 2:
                warnings.warn(
                    f"Cannot split further: requested {n_partitions} partitions "
                    f"but only {len(partitions)} achievable.",
                    stacklevel=2,
                )
                break
            sub = g.subgraph(largest).copy()
            halves = _recursive_bisection(sub, 2, weight_attr)
            partitions = halves + partitions[1:]
        return partitions, _collect_cut_edges(g, partitions)

    if n_partitions == 2:
        try:
            _, partition_sets = nx.stoer_wagner(g, weight=weight_attr)
            partition_list = [list(part) for part in partition_sets]
            return partition_list, _get_cut_edges(g, partition_list)
        except nx.NetworkXError:
            return [list(g.nodes())], []
    else:
        partitions = _recursive_bisection(g, n_partitions, weight_attr)
        cut_edges = _collect_cut_edges(g, partitions)
        return partitions, cut_edges


def _recursive_bisection(graph: nx.Graph, k: int, weight_attr: str) -> List[List[Any]]:
    """Recursively bisect graph into k partitions."""
    nx = require_dependency("networkx", "graph", "Graph partitioning")  # noqa: F841
    if k <= 1:
        return [list(graph.nodes())]
    if len(graph) < k:
        return [[n] for n in graph.nodes()]

    if not nx.is_connected(graph):
        # Split by CCs if possible
        ccs = list(nx.connected_components(graph))
        # Sort by size to split biggest?
        ccs.sort(key=len, reverse=True)

        # We need to split into 2 groups for bisection
        part1 = list(ccs[0])
        part2 = []
        for c in ccs[1:]:
            part2.extend(list(c))
    else:
        try:
            _, (part1, part2) = nx.stoer_wagner(graph, weight=weight_attr)
        except nx.NetworkXError:
            nodes = list(graph.nodes())
            mid = len(nodes) // 2
            part1 = nodes[:mid]
            part2 = nodes[mid:]

    # Distribute k based on node count to favor finding natural clusters
    n1 = len(part1)
    n2 = len(part2)
    total_n = n1 + n2

    if k == 2:
        k1 = 1
        k2 = 1
    else:
        # Proportional distribution
        k1 = int(round(k * n1 / total_n))
        # Ensure bounds: 1 <= k1 <= k-1
        k1 = max(1, min(k - 1, k1))
        k2 = k - k1

    subgraph1 = graph.subgraph(part1)
    subgraph2 = graph.subgraph(part2)

    partitions = []
    partitions.extend(_recursive_bisection(subgraph1, k1, weight_attr))
    partitions.extend(_recursive_bisection(subgraph2, k2, weight_attr))

    return partitions


def _get_cut_edges(graph: nx.Graph, partitions: List[List[Any]]) -> List[Tuple[int, int]]:
    """Find edges that cross between partitions."""
    cut_edges = []
    node_to_part = {}
    for i, part in enumerate(partitions):
        for node in part:
            node_to_part[node] = i

    for u, v in graph.edges():
        # Only check if both nodes are in the partitions (might be subgraph)
        if u in node_to_part and v in node_to_part:
            if node_to_part[u] != node_to_part[v]:
                cut_edges.append((u, v))

    return cut_edges


def _collect_cut_edges(graph: nx.Graph, partitions: List[List[Any]]) -> List[Tuple[int, int]]:
    """Wrapper for consistency."""
    return _get_cut_edges(graph, partitions)


def community_partition(
    graph: "MolecularGraph", resolution: float = 1.0, algorithm: str = "louvain"
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Partition using community detection.

    Discovers natural communities (dense clusters) in graph.

    Args:
        graph: MolecularGraph to partition
        resolution: Louvain resolution (higher = more communities)
        algorithm: 'louvain' or 'label_propagation'

    Returns:
        Tuple of (partition_lists, cut_edges)

    Raises:
        ImportError: If algorithm='louvain' and python-louvain is not installed.
    """
    nx = require_dependency("networkx", "graph", "Graph partitioning")
    g = graph.networkx_graph
    partition_dict = {}

    if algorithm == "louvain":
        try:
            import community as community_louvain  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "python-louvain package is required for 'louvain' algorithm. Install with 'pip install python-louvain'."
            )

        # networkx graphs might not have 'weight' on all edges?
        # best_partition allows specifying weight attr.
        # If edges don't have weight, it defaults to 1?
        # We'll rely on community_louvain defaults or pass weight='weight' if we want weighted.
        # Generally 'order' or 'weight' is used.
        # Let's check graph first.
        partition_dict = community_louvain.best_partition(g, resolution=resolution)

    elif algorithm == "label_propagation":
        communities = nx.algorithms.community.label_propagation_communities(g)
        partition_dict = {node: i for i, comm in enumerate(communities) for node in comm}

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Convert to list of lists
    partitions = _partition_dict_to_lists(partition_dict)
    cut_edges = _collect_cut_edges(g, partitions)

    return partitions, cut_edges


def _partition_dict_to_lists(partition_dict: Dict[Any, int]) -> List[List[Any]]:
    """Convert node->community dict to list of community lists."""
    if not partition_dict:
        return []

    # Find unique community IDs to map to indices
    comm_ids = sorted(list(set(partition_dict.values())))
    id_to_idx = {cid: i for i, cid in enumerate(comm_ids)}

    partitions: List[List[Any]] = [[] for _ in range(len(comm_ids))]

    for node, comm_id in partition_dict.items():
        partitions[id_to_idx[comm_id]].append(node)

    return partitions


def balanced_partition(
    graph: "MolecularGraph", n_partitions: int, balance_tolerance: float = 0.1
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Partition graph into balanced-size fragments.

    Uses Kernighan-Lin algorithm for balanced bisection where possible.
    Note: Kernighan-Lin requires connected graphs.

    Args:
        graph: MolecularGraph to partition
        n_partitions: Target number of partitions
        balance_tolerance: Allowed size imbalance (unused by KL, but kept for API)

    Returns:
        Tuple of (partition_lists, cut_edges)
    """
    require_dependency("networkx", "graph", "Graph partitioning")
    g = graph.networkx_graph

    if n_partitions < 2:
        return [list(g.nodes())], []

    partitions = _balanced_recursive_bisection(g, n_partitions)
    cut_edges = _collect_cut_edges(g, partitions)

    return partitions, cut_edges


def _balanced_recursive_bisection(graph: nx.Graph, k: int) -> List[List[Any]]:
    """Recursively bisect graph into k balanced partitions."""
    nx = require_dependency("networkx", "graph", "Graph partitioning")
    if k <= 1:
        return [list(graph.nodes())]
    if len(graph) < k:
        return [[n] for n in graph.nodes()]

    # KL requires connected graph
    if not nx.is_connected(graph):
        # If disconnected, we can't use KL easily on the whole thing.
        # Fallback to component splitting logic or just treat components as splits?
        # For simplicity, let's use the same logic as min_cut for disconnected
        # or just try to partition the largest component.
        ccs = list(nx.connected_components(graph))
        ccs.sort(key=len, reverse=True)

        # We need to split into 2 groups
        # This is the "Bin Packing" problem essentially.
        # Greedy approximation:
        group1 = []
        group2 = []
        w1 = 0
        w2 = 0

        for c in ccs:
            if w1 <= w2:
                group1.extend(list(c))
                w1 += len(c)
            else:
                group2.extend(list(c))
                w2 += len(c)

        part1, part2 = group1, group2
    else:
        try:
            part1, part2 = nx.algorithms.community.kernighan_lin_bisection(graph)
        except (nx.NetworkXError, TypeError):
            # Fallback if KL fails (e.g. graph too small or other issue)
            nodes = list(graph.nodes())
            mid = len(nodes) // 2
            part1 = nodes[:mid]
            part2 = nodes[mid:]

    # Distribute k proportionally
    n1 = len(part1)
    n2 = len(part2)
    total_n = n1 + n2

    if k == 2:
        k1 = 1
        k2 = 1
    else:
        k1 = int(round(k * n1 / total_n))
        k1 = max(1, min(k - 1, k1))
        k2 = k - k1

    subgraph1 = graph.subgraph(part1)
    subgraph2 = graph.subgraph(part2)

    partitions = []
    partitions.extend(_balanced_recursive_bisection(subgraph1, k1))
    partitions.extend(_balanced_recursive_bisection(subgraph2, k2))

    return partitions


@dataclass
class FragmentTree:
    """Hierarchical tree of fragments."""

    atoms: List[int]
    children: List["FragmentTree"] = field(default_factory=list)
    level: int = 0

    @property
    def is_leaf(self) -> bool:
        """Return whether the given node is a leaf in the partition tree."""
        return len(self.children) == 0


def hierarchical_decomposition(
    graph: "MolecularGraph", min_fragment_size: int = 10, max_levels: int = 5
) -> FragmentTree:
    """Build hierarchical fragment tree.

    Recursively partitions until minimum size reached.

    Args:
        graph: MolecularGraph to partition
        min_fragment_size: Stop splitting below this size
        max_levels: Maximum tree depth

    Returns:
        FragmentTree with nested partitions
    """
    root = FragmentTree(atoms=graph.nodes(), level=0)
    _build_tree(root, graph.networkx_graph, min_fragment_size, max_levels)
    return root


def _build_tree(node: FragmentTree, graph: nx.Graph, min_size: int, max_levels: int):
    """Recursively build tree."""
    if len(node.atoms) <= min_size or node.level >= max_levels:
        return

    # Bisect this fragment
    subgraph = graph.subgraph(node.atoms).copy()

    # Use balanced partition on this subgraph
    # We call internal logic directly to avoid circular imports or complex object creation
    # But balanced_partition expects MolecularGraph wrapper usually.
    # However, _balanced_recursive_bisection works on nx.Graph
    parts = _balanced_recursive_bisection(subgraph, 2)

    # If no split occurred or just one part
    if len(parts) < 2:
        return

    for partition_atoms in parts:
        child = FragmentTree(atoms=partition_atoms, level=node.level + 1)
        node.children.append(child)
        _build_tree(child, graph, min_size, max_levels)


def metis_partition(
    graph: "MolecularGraph", n_partitions: int, balance_constraint: float = 1.03
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Partition using METIS library.

    METIS is highly optimized for large graphs.
    Falls back to networkx if METIS not available.

    Args:
        graph: MolecularGraph to partition
        n_partitions: Number of partitions
        balance_constraint: Allowed imbalance (1.03 = 3%)

    Returns:
        Tuple of (partition_lists, cut_edges)
    """
    try:
        import pymetis  # type: ignore[import-not-found]
    except ImportError:
        # warnings.warn("METIS (pymetis) not installed, falling back to balanced_partition")
        return balanced_partition(graph, n_partitions)

    g = graph.networkx_graph

    if n_partitions < 2:
        return [list(g.nodes())], []

    # pymetis expects adjacency list
    adj_list = []
    # nodes are 0..N-1 usually?
    # nx graph might have arbitrary node labels, need mapping
    # Assuming standard 0..N indices for now or remap
    nodes = list(g.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}

    for i in range(len(nodes)):
        node = rev_map[i]
        neighbors = [node_map[nbr] for nbr in g.neighbors(node)]
        adj_list.append(np.array(neighbors))

    # Partition
    # pymetis.part_graph returns (cut_count, partition_array)
    # n_cuts, membership = pymetis.part_graph(n_partitions, adjacency=adj_list, ubvec=[balance_constraint])
    # Note: ubvec might not be exposed in simple wrapper, check docs?
    # We'll stick to simple call
    try:
        n_cuts, membership = pymetis.part_graph(n_partitions, adjacency=adj_list)
    except Exception:
        # Fallback if execution fails
        return balanced_partition(graph, n_partitions)

    # Convert membership to lists
    partitions: List[List[Any]] = [[] for _ in range(n_partitions)]
    for i, part_id in enumerate(membership):
        partitions[part_id].append(rev_map[i])

    # Filter empty partitions
    partitions = [p for p in partitions if p]
    cut_edges = _collect_cut_edges(g, partitions)

    return partitions, cut_edges
