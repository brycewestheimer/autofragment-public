# Graph Partitioning for Molecular Fragmentation

## Introduction

Graph partitioning algorithms divide a molecular graph into disjoint subsets (fragments) while minimizing the weight of edges between partitions. These methods are fundamental to molecular fragmentation, where the graph representation naturally captures atomic connectivity.

## Molecular Graph Representation

A molecule is represented as a weighted graph $G = (V, E, w)$ where:
- $V = \{v_1, \ldots, v_n\}$: vertices (atoms)
- $E \subseteq V \times V$: edges (bonds)
- $w: E \to \mathbb{R}^+$: edge weights (bond strengths)

Edge weights can encode:
- **Bond order**: single (1.0), double (2.0), triple (3.0), aromatic (1.5)
- **Bond energy**: energy cost of breaking the bond
- **Interaction strength**: electrostatic or van der Waals interactions

## Minimum Cut Problem

### Definition

A **cut** $(S, \bar{S})$ partitions vertices into two disjoint sets where $S \cup \bar{S} = V$.

The **cut weight** is the sum of edge weights crossing the partition:

$$
\text{cut}(S, \bar{S}) = \sum_{u \in S, v \in \bar{S}} w(u, v)
$$

The **minimum cut** minimizes this value:

$$
(S^*, \bar{S}^*) = \arg\min_{S \subset V} \text{cut}(S, \bar{S})
$$

### Stoer-Wagner Algorithm

Finds the global minimum cut in $O(|V||E| + |V|^2 \log|V|)$ time:

1. Start with arbitrary vertex $a$
2. Grow a maximum adjacency ordering from $a$
3. The last two vertices $(s, t)$ define a cut-of-the-phase
4. Contract $s$ and $t$ into a single vertex
5. Repeat until two vertices remain
6. Return minimum cut found across all phases

### Limitation: Trivial Cuts

Minimum cut tends to isolate single vertices (especially terminal atoms like hydrogen). For molecular fragmentation, we need **balanced** partitions.

## Balanced Partitioning

### Ratio Cut

Normalize the cut by partition sizes:

$$
\text{RatioCut}(S, \bar{S}) = \frac{\text{cut}(S, \bar{S})}{|S|} + \frac{\text{cut}(S, \bar{S})}{|\bar{S}|}
$$

This penalizes highly imbalanced partitions.

### Normalized Cut (NCut)

Normalize by the **volume** (total edge weight) of each partition:

$$
\text{NCut}(S, \bar{S}) = \frac{\text{cut}(S, \bar{S})}{\text{vol}(S)} + \frac{\text{cut}(S, \bar{S})}{\text{vol}(\bar{S})}
$$

where:

$$
\text{vol}(S) = \sum_{i \in S} d_i = \sum_{i \in S} \sum_{j} w_{ij}
$$

NCut is equivalent to finding the second smallest eigenvector of the normalized Laplacian (spectral clustering).

### Kernighan-Lin Algorithm

Iterative improvement algorithm for balanced bipartitioning:

1. **Initialize**: Split vertices into two equal-sized sets $A$ and $B$
2. **Compute gains**: For each pair $(a, b)$ with $a \in A$, $b \in B$:
   $$
   g(a, b) = D_a + D_b - 2w(a, b)
   $$
   where $D_a = \sum_{v \in B} w(a, v) - \sum_{v \in A} w(a, v)$ is the external - internal cost
3. **Select best swap**: Find $(a^*, b^*)$ maximizing $g$
4. **Lock and update**: Mark swapped vertices, update gains
5. **Repeat**: Until all vertices considered
6. **Apply prefix**: Keep swaps up to maximum cumulative gain

**Complexity**: $O(n^2 \log n)$ per pass, typically converges in few passes.

## Community Detection

Community detection finds densely connected groups without requiring balance constraints.

### Modularity

Modularity measures the quality of a partition by comparing edge density within communities to a random graph:

$$
Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

where:
- $A_{ij}$: adjacency matrix entry
- $k_i = \sum_j A_{ij}$: degree of vertex $i$
- $m = \frac{1}{2}\sum_{ij} A_{ij}$: total edge weight
- $c_i$: community assignment of vertex $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$, else $0$

Modularity ranges from $-0.5$ to $1$, with higher values indicating stronger community structure.

### Louvain Algorithm

Fast modularity optimization with hierarchical aggregation:

**Phase 1 (Local moves)**:
1. For each vertex $i$ in random order
2. Compute modularity gain from moving $i$ to each neighbor's community:
   $$
   \Delta Q = \left[ \frac{\Sigma_{in} + k_{i,in}}{2m} - \left(\frac{\Sigma_{tot} + k_i}{2m}\right)^2 \right] - \left[ \frac{\Sigma_{in}}{2m} - \left(\frac{\Sigma_{tot}}{2m}\right)^2 - \left(\frac{k_i}{2m}\right)^2 \right]
   $$
   where $\Sigma_{in}$ = sum of weights inside community, $\Sigma_{tot}$ = total weight of community, $k_{i,in}$ = sum of weights from $i$ to community
3. Move $i$ to community with maximum positive $\Delta Q$
4. Repeat until no improvement

**Phase 2 (Aggregation)**:
1. Build new graph where each community becomes a super-vertex
2. Edge weights between super-vertices = sum of edges between constituent communities
3. Self-loops = internal edges of original community

**Repeat**: Apply Phase 1 to aggregated graph until convergence.

**Complexity**: $O(n \log n)$ for sparse graphs, making it suitable for large molecules.

### Resolution Parameter

Modularity has a resolution limit—it may merge small communities. Introduce resolution parameter $\gamma$:

$$
Q_\gamma = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \gamma\frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

- $\gamma > 1$: Favor smaller communities (more fragments)
- $\gamma < 1$: Favor larger communities (fewer fragments)

## METIS Algorithm

METIS is an industry-standard multilevel graph partitioning library optimized for large graphs.

### Multilevel Paradigm

**1. Coarsening Phase**
Progressively collapse vertices into coarser graphs:

- **Heavy Edge Matching (HEM)**: Match vertices connected by heaviest edges
- **Sorted Heavy Edge Matching (SHEM)**: Process vertices in decreasing degree order

After $\ell$ coarsening levels: $|V_\ell| \ll |V_0|$

**2. Initial Partitioning**
Apply exact or heuristic algorithm to small coarsened graph. Common choices:
- Greedy bisection
- Spectral partitioning
- Kernighan-Lin

**3. Uncoarsening/Refinement**
Project partition back to finer levels, refining at each step:

- **Boundary refinement**: Only consider moving vertices at partition boundary
- **FM refinement**: Fiduccia-Mattheyses variant optimized for moves

### METIS for Molecules

METIS partitions into $k$ balanced parts minimizing edge cut. For molecules:

```python
import pymetis

# Molecule as adjacency list
adjacency = [[1, 2], [0, 2, 3], [0, 1], [1, 4], [3]]

# Partition into 2 fragments
n_cuts, membership = pymetis.part_graph(2, adjacency)

# membership[i] = fragment assignment for atom i
```

**Advantages**:
- Very fast: $O(|E|)$ time complexity
- Produces balanced partitions
- Minimal edge cut (bond breaking)

**Considerations**:
- Requires installation of `pymetis`
- Balance constraint may conflict with chemical constraints

## k-way Partitioning

### Direct k-way

Partition into $k > 2$ parts directly:

$$
\text{minimize} \quad \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} \text{cut}(V_i, V_j)
$$

subject to balance constraints:

$$
|V_i| \leq (1 + \epsilon) \frac{|V|}{k}, \quad \forall i
$$

### Recursive Bisection

Alternative approach:
1. Bisect graph into two parts
2. Recursively bisect each part
3. Continue until $k$ parts obtained

For $k = 2^p$, requires $k-1$ cuts.

## Application to Molecular Fragmentation

### Weight Functions for Chemistry

Design edge weights that encode chemical knowledge:

**Bond energy weight**:
$$
w(u, v) = E_{\text{bond}}(u, v)
$$
Penalizes breaking strong bonds.

**Aromaticity weight**:
$$
w(u, v) = \begin{cases}
\infty & \text{if bond is aromatic} \\
E_{\text{bond}} & \text{otherwise}
\end{cases}
$$
Prevents breaking aromatic systems.

**Functional group weight**:
$$
w(u, v) = \begin{cases}
\infty & \text{if bond is within functional group} \\
E_{\text{bond}} \cdot f(\text{rotatable}) & \text{otherwise}
\end{cases}
$$

### Comparison of Methods

| Method | Complexity | Balance | Quality | Best For |
|--------|-----------|---------|---------|----------|
| Min-cut | $O(VE + V^2 \log V)$ | No | Exact | Bipartition |
| Kernighan-Lin | $O(n^2 \log n)$ | Strict | Good | Balanced bipartition |
| Louvain | $O(n \log n)$ | No | Good | Natural clusters |
| METIS | $O(E)$ | Yes | Excellent | Large molecules |

## References

1. Stoer, M., & Wagner, F. (1997). A simple min-cut algorithm. *JACM*, 44(4), 585-591.
2. Kernighan, B. W., & Lin, S. (1970). An efficient heuristic procedure for partitioning graphs. *Bell System Technical Journal*, 49(2), 291-307.
3. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *JSTAT*, P10008.
4. Karypis, G., & Kumar, V. (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. *SIAM Journal on Scientific Computing*, 20(1), 359-392.
5. Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. *IEEE TPAMI*, 22(8), 888-905.
