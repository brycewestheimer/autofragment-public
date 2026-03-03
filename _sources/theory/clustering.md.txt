# Clustering Algorithms for Molecular Fragmentation

## Introduction

Clustering algorithms partition atoms or molecular units into groups based on spatial proximity, connectivity, or chemical similarity. In molecular fragmentation, these algorithms help identify natural groupings that can be treated as computational fragments.

AutoFragment implements several clustering approaches, each with distinct advantages:

- **K-means**: Fast, scalable, best for roughly spherical clusters
- **Spectral clustering**: Captures complex connectivity patterns
- **Hierarchical methods**: Produces nested fragment hierarchies

## K-Means Clustering

### Objective Function

Given $n$ atoms and $k$ target fragments, k-means minimizes the within-cluster sum of squares (WCSS):

$$
J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2
$$

where:
- $C_i$ is the set of atoms in cluster $i$
- $\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \mathbf{x}$ is the centroid of cluster $i$
- $\|\cdot\|$ denotes the Euclidean distance

### Algorithm (Lloyd's Method)

1. **Initialize**: Choose $k$ initial centroids $\boldsymbol{\mu}_1^{(0)}, \ldots, \boldsymbol{\mu}_k^{(0)}$
2. **Assign**: For each atom $\mathbf{x}_j$, assign to nearest centroid:
   $$
   c_j^{(t)} = \arg\min_{i} \|\mathbf{x}_j - \boldsymbol{\mu}_i^{(t)}\|^2
   $$
3. **Update**: Recompute centroids:
   $$
   \boldsymbol{\mu}_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{\mathbf{x}_j \in C_i^{(t)}} \mathbf{x}_j
   $$
4. **Repeat**: Until convergence ($c_j^{(t+1)} = c_j^{(t)}$ for all $j$)

### Initialization Strategies

#### Random Initialization

Select $k$ atoms uniformly at random as initial centroids. Simple but can lead to poor local minima.

#### K-means++ (Recommended)

Weighted random selection that favors spread-out centroids:

1. Choose first centroid uniformly at random
2. For each atom $\mathbf{x}_j$, compute $D(\mathbf{x}_j)$ = distance to nearest centroid
3. Choose next centroid with probability:
   $$
   P(\mathbf{x}_j) = \frac{D(\mathbf{x}_j)^2}{\sum_{\ell} D(\mathbf{x}_\ell)^2}
   $$
4. Repeat until $k$ centroids chosen

K-means++ provides an expected approximation ratio of $O(\log k)$ compared to optimal clustering.

### Convergence Properties

- **Guaranteed convergence**: The objective $J$ decreases monotonically and is bounded below
- **Local minimum**: Converges to local (not necessarily global) minimum
- **Recommendation**: Run multiple restarts (typically 10-20) and select best solution

### Complexity Analysis

| Operation | Time Complexity |
|-----------|-----------------|
| One iteration | $O(nkd)$ |
| Total (t iterations) | $O(tnkd)$ |
| K-means++ init | $O(nk)$ |

where $n$ = number of atoms, $k$ = number of clusters, $d$ = dimension (typically 3 for coordinates).

## Spectral Clustering

Spectral clustering uses eigenvalues of a graph Laplacian matrix to embed atoms in a space where k-means works well, even for non-convex clusters.

### Molecular Graph Representation

Represent the molecule as a weighted graph $G = (V, E, W)$:
- Vertices $V$: atoms
- Edges $E$: bonds or proximity-based connections
- Weights $W$: bond strengths, distances, or interaction energies

### Affinity Matrix

The affinity (or similarity) matrix $A \in \mathbb{R}^{n \times n}$ encodes pairwise relationships:

$$
A_{ij} = \begin{cases}
w_{ij} & \text{if atoms } i, j \text{ are connected} \\
0 & \text{otherwise}
\end{cases}
$$

For distance-based affinity, use Gaussian kernel:

$$
A_{ij} = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)
$$

### Degree Matrix

The degree matrix $D \in \mathbb{R}^{n \times n}$ is diagonal:

$$
D_{ii} = \sum_{j=1}^{n} A_{ij} = d_i
$$

### Graph Laplacian

The **unnormalized Laplacian**:

$$
L = D - A
$$

Properties:
- Symmetric, positive semi-definite
- Smallest eigenvalue is 0 with eigenvector $\mathbf{1}$
- Number of 0 eigenvalues = number of connected components

### Normalized Laplacian

The **symmetric normalized Laplacian**:

$$
L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
$$

The **random walk Laplacian**:

$$
L_{\text{rw}} = D^{-1} L = I - D^{-1} A
$$

### Spectral Clustering Algorithm

1. Compute the first $k$ eigenvectors of $L_{\text{sym}}$ (smallest eigenvalues):
   $$
   L_{\text{sym}} \mathbf{u}_i = \lambda_i \mathbf{u}_i, \quad i = 1, \ldots, k
   $$

2. Form matrix $U \in \mathbb{R}^{n \times k}$ with eigenvectors as columns

3. Normalize rows of $U$ to unit length:
   $$
   \tilde{\mathbf{u}}_j = \frac{\mathbf{u}_j}{\|\mathbf{u}_j\|}
   $$

4. Apply k-means to the rows of $\tilde{U}$ to obtain clusters

### Spectral Gap Heuristic

The optimal number of clusters can be estimated from the **spectral gap**:

$$
k^* = \arg\max_k (\lambda_{k+1} - \lambda_k)
$$

A large gap indicates natural cluster structure.

### Complexity Analysis

| Operation | Time Complexity |
|-----------|-----------------|
| Laplacian construction | $O(n^2)$ or $O(|E|)$ |
| Eigendecomposition | $O(n^3)$ or $O(kn^2)$ for $k$ eigenvectors |
| K-means on embeddings | $O(tnk^2)$ |

## Hierarchical Clustering

### Agglomerative Clustering

Bottom-up approach that builds a hierarchy (dendrogram):

1. Start with each atom as its own cluster
2. Merge the two closest clusters
3. Repeat until single cluster remains

### Linkage Criteria

Distance between clusters $C_i$ and $C_j$:

**Single linkage** (minimum):
$$
d(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
$$

**Complete linkage** (maximum):
$$
d(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
$$

**Average linkage** (UPGMA):
$$
d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
$$

**Ward's method** (minimize variance):
$$
d(C_i, C_j) = \frac{|C_i||C_j|}{|C_i| + |C_j|} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|^2
$$

### Choosing Number of Clusters

Cut the dendrogram at a chosen height or number of clusters. For molecular fragmentation, Ward's method often produces the most balanced fragments.

## Application to Molecular Fragmentation

### Distance Metrics for Molecules

Beyond Euclidean distance, consider:

**Through-space distance**: 
$$
d_{\text{space}}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|
$$

**Through-bond distance**: 
$$
d_{\text{bond}}(i, j) = \text{shortest path in molecular graph}
$$

**Weighted distance**:
$$
d_{\text{combined}}(i, j) = \alpha \cdot d_{\text{space}}(i, j) + (1-\alpha) \cdot d_{\text{bond}}(i, j)
$$

### Constraint-Aware Clustering

Incorporate chemical constraints:
- **Must-link**: Atoms that must be in the same fragment
- **Cannot-link**: Atoms that must be in different fragments

These constraints can be incorporated into spectral clustering by modifying the affinity matrix.

## References

1. Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027-1035.
3. Von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416.
4. Ng, A. Y., Jordan, M. I., & Weiss, Y. (2001). On spectral clustering: Analysis and an algorithm. *NIPS*, 849-856.
