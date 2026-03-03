# Fragmentation Algorithms Guide

AutoFragment provides a suite of algorithms for molecular fragmentation, ranging from simple clustering to advanced graph partitioning. This guide covers how to choose, configure, and combine these algorithms for optimal results.

## Overview

Fragmentation in AutoFragment typically follows one of two paradigms:

1.  **Clustering**: Spatial-based partitioning using atom/molecule coordinates.
2.  **Graph Partitioning**: Topology-based partitioning using bond connectivity.

## Clustering Algorithms

Clustering is fast and effective for systems without a complex covalent network, such as water clusters or solvent boxes.

```python
from autofragment import MolecularPartitioner

# Use standard k-means (default)
partitioner = MolecularPartitioner(
    n_fragments=10,
    method="kmeans"
)
tree = partitioner.partition(system)
```

Available clustering methods:
- `kmeans`: Standard k-means (default, no extra dependencies).
- `kmeans_constrained`: K-means with size constraints (requires `pip install autofragment[balanced]`).
- `spectral`: Spectral clustering on the adjacency matrix.
- `agglomerative`: Hierarchical clustering.
- `gmm`: Gaussian Mixture Models.
- `birch`: BIRCH incremental clustering.
- `geom_planes`: Geometric plane-based partitioning.

### Topology-Assisted Refinement

For systems where pure spatial clustering is ambiguous, you can apply topology-based post-refinement to improve chemical locality.

```python
from autofragment import MolecularPartitioner

partitioner = MolecularPartitioner(
    n_fragments=8,
    method="kmeans",
    topology_refine=True,
    topology_mode="graph",      # or "euclidean"
    topology_hops=1,
    topology_layers=1,
    topology_k_per_layer=6,
    topology_bond_policy="infer" # or "strict"
)

tree = partitioner.partition(system)
```

This refinement keeps default behavior unchanged when `topology_refine=False`.

## K-Means Seeding Strategies

AutoFragment provides pluggable initialization strategies for k-means clustering,
giving better control over initial centroid placement:

```python
from autofragment import MolecularPartitioner

# Use PCA-based seeding
partitioner = MolecularPartitioner(
    n_fragments=4, method="kmeans", init_strategy="pca"
)
```

Available strategies:

| Strategy | Description |
|----------|-------------|
| `"halfplane"` | Angular sectors in the PCA plane |
| `"pca"` | Binning along the first principal component |
| `"axis"` | Binning along a Cartesian axis (auto-detected or explicit) |
| `"radial"` | XY angular sectors around the center of mass |

For the `axis` strategy, you can specify which axis to use:

```python
partitioner = MolecularPartitioner(
    n_fragments=4, method="kmeans",
    init_strategy={"strategy": "axis", "axis": "z"}
)
```

Seeding strategies also work with `kmeans_constrained`. They are silently ignored
for methods that don't support custom initialization (agglomerative, spectral, etc.).

### Direct Seeding API

```python
from autofragment.algorithms.seeding import compute_seeds, SEEDING_STRATEGIES

# Compute seed centroids directly
seeds = compute_seeds(centroids, n_clusters=4, strategy="pca")
# seeds.shape == (4, 3)
```

## Graph Partitioning

Graph-based methods use the covalent connectivity of the molecule to find natural "cut points," avoiding breaking strong bonds or rings.

### Min-Cut (Stoer-Wagner)

Finds the global minimum cut in the molecular graph.

```python
from autofragment.algorithms.graph_partition import min_cut_partition

# Find the easiest way to split a molecule in two
cut_edges, partition1, partition2 = min_cut_partition(molecular_graph)
```

### Balanced Partitioning (Kernighan-Lin)

Recursive bisection that prioritizes equal-sized fragments while minimizing the weight of cut bonds.

```python
from autofragment.algorithms.graph_partition import balanced_partition

# Split into 4 balanced fragments
groups = balanced_partition(molecular_graph, n_fragments=4)
```

### Community Detection (Louvain)

Finds "communities" or natural clusters based on modularity. Excellent for discovery-based fragmentation.

```python
from autofragment.algorithms.graph_partition import community_partition

# Discover natural fragments
communities = community_partition(molecular_graph, algorithm="louvain")
```

## Scoring

The scoring system allows you to quantify the quality of a fragmentation:

```python
from autofragment.algorithms.scoring import FragmentationScore

score = FragmentationScore(
    bond_penalty=2.5,
    size_variance=0.1,
    interface_score=0.8,
    integrity_score=0.95,
)
print(f"Total Score: {score.total()}")
print(f"Bond Penalty: {score.bond_penalty}")
print(f"Size Variance: {score.size_variance}")
```

Preset scoring weights for common use cases:

```python
from autofragment.algorithms.scoring import ScoringWeights

# For FMO calculations
weights = ScoringWeights.for_fmo()

# For many-body expansion
weights = ScoringWeights.for_mbe()
```

## Choosing the Right Algorithm

| System Type | Recommended Method | Why? |
|-------------|--------------------|------|
| Water Clusters | `kmeans` or `kmeans_constrained` | Spatial proximity is key; uniform sizes. |
| Large Water Clusters | `kmeans` with `tiers=2` | Hierarchical grouping for HMBE workflows. |
| Proteins | `bio` (BioPartitioner) | Respects biological units and capping. |
| Large Molecules | `balanced_partition` | Maintains chemical connectivity while balancing load. |
| Disordered Solids | `community_partition` | Finds natural clusters in complex networks. |

See the [Theory Section](../theory/scoring.md) for the mathematical details behind these algorithms.
