# Many-Body Expansion (MBE) Theory

## Introduction

The many-body expansion (MBE) is a systematic approach to compute the energy of a molecular system as a sum of fragment energies and their interactions. It provides a rigorous framework for fragment-based quantum chemistry methods, enabling accurate calculations on systems too large for conventional approaches.

## Mathematical Foundation

### Total Energy Expression

The exact energy of a system of $N$ fragments can be written as:

$$
E_{\text{total}} = \sum_i E_i + \sum_{i<j} \Delta E_{ij} + \sum_{i<j<k} \Delta E_{ijk} + \cdots + \Delta E_{12\ldots N}
$$

This expansion is **exact** when carried to $N$-body terms, but practical calculations truncate at lower orders.

### n-Body Interaction Energies

#### One-Body Terms (Monomers)

The one-body terms are simply fragment energies:

$$
E^{(1)} = \sum_{i=1}^{N} E_i
$$

where $E_i$ is the quantum mechanical energy of fragment $i$ in isolation.

#### Two-Body Terms (Dimers)

Two-body corrections capture pairwise interactions:

$$
\Delta E_{ij} = E_{ij} - E_i - E_j
$$

This is the **interaction energy** between fragments $i$ and $j$. The two-body contribution to total energy:

$$
E^{(2)} = \sum_{i<j} \Delta E_{ij}
$$

#### Three-Body Terms (Trimers)

Three-body corrections capture **non-additive** effects:

$$
\Delta E_{ijk} = E_{ijk} - E_{ij} - E_{jk} - E_{ik} + E_i + E_j + E_k
$$

Equivalently:

$$
\Delta E_{ijk} = E_{ijk} - \Delta E_{ij} - \Delta E_{jk} - \Delta E_{ik} - E_i - E_j - E_k
$$

Three-body contribution:

$$
E^{(3)} = \sum_{i<j<k} \Delta E_{ijk}
$$

#### Higher-Order Terms

The pattern continues for four-body and higher:

$$
\Delta E_{ijk\ell} = E_{ijk\ell} - \sum_{\text{trimers}} \Delta E - \sum_{\text{dimers}} \Delta E - \sum_{\text{monomers}} E
$$

### Truncated MBE

In practice, MBE is truncated at order $n$:

$$
E_{\text{MBE}(n)} = \sum_i E_i + \sum_{i<j} \Delta E_{ij} + \cdots + \sum_{i_1 < \cdots < i_n} \Delta E_{i_1 \cdots i_n}
$$

Common truncation levels:

| Order | Name | Formula |
|-------|------|---------|
| MBE(1) | Monomer sum | $\sum_i E_i$ |
| MBE(2) | Pairwise | $\sum_i E_i + \sum_{i<j} \Delta E_{ij}$ |
| MBE(3) | Three-body | $\sum_i E_i + \sum_{i<j} \Delta E_{ij} + \sum_{i<j<k} \Delta E_{ijk}$ |

## Convergence Properties

### Physical Basis

The MBE converges because many-body interactions decay with distance:

- **Two-body**: Electrostatics $\sim r^{-1}$, dispersion $\sim r^{-6}$
- **Three-body**: Axilrod-Teller $\sim r^{-9}$, induction $\sim r^{-6}$
- **Higher-order**: Faster decay, typically negligible

### Accuracy Guidelines

Typical errors for molecular clusters (kcal/mol):

| System Type | MBE(2) Error | MBE(3) Error |
|-------------|--------------|--------------|
| Water clusters | 1-5 | 0.1-0.5 |
| Noble gas clusters | 0.1-0.5 | < 0.05 |
| Ionic systems | 5-20 | 1-3 |
| π-stacking | 2-5 | 0.3-1 |

### Factors Affecting Convergence

1. **Fragment size**: Larger fragments → faster convergence
2. **Basis set**: Larger basis → better convergence
3. **Interaction type**: Electrostatic > dispersion > charge transfer
4. **Geometry**: Compact geometries converge slower

### Problematic Cases

MBE converges slowly for:
- Strong charge transfer
- Significant covalent character across boundaries
- Extended conjugation
- Metal clusters

In these cases, consider:
- Larger fragments
- Higher-order terms
- Embedding approaches (EFP, QM/MM)

## Computational Scaling

### Number of Calculations

| Order | Number of Calculations | Scaling |
|-------|----------------------|---------|
| 1-body | $N$ | $O(N)$ |
| 2-body | $\binom{N}{2} = \frac{N(N-1)}{2}$ | $O(N^2)$ |
| 3-body | $\binom{N}{3} = \frac{N(N-1)(N-2)}{6}$ | $O(N^3)$ |
| n-body | $\binom{N}{n}$ | $O(N^n)$ |

### Distance Screening

Reduce calculations by screening distant pairs:

$$
\Delta E_{ij} \approx 0 \quad \text{if} \quad R_{ij} > R_{\text{cutoff}}
$$

Typical cutoffs:
- Two-body: 10-15 Å (electrostatics), 6-8 Å (exchange)
- Three-body: 6-8 Å

With screening, effective scaling drops to $O(N)$ for large systems.

### Parallelization

MBE is **embarrassingly parallel**:
- All $n$-body calculations at a given order are independent
- Natural fit for distributed computing
- Near-linear parallel speedup

## Hierarchy of Methods

### Electrostatically Embedded MBE (EE-MBE)

Include electrostatic environment in fragment calculations:

$$
E_i^{\text{embed}} = E_i^{\text{QM}} + \sum_{j \neq i} E_{ij}^{\text{point-charge}}
$$

Improves two-body truncation by capturing polarization.

### Systematic Molecular Fragmentation (SMF)

Uses graph-based fragmentation with overlapping fragments. Energy reconstructed via inclusion-exclusion.

### Generalized MBE (GMBE)

Allows overlapping fragments with proper overcounting corrections.

## Implementation Notes

### Fragmentation with autofragment

autofragment handles the fragmentation step of an MBE workflow — partitioning a
molecular system into fragments and writing QC input files. The n-body expansion
and energy assembly are performed externally after running the individual QC
calculations.

```python
import autofragment as af
from autofragment.partitioners import MolecularPartitioner
from autofragment.io.writers import write_gamess_fmo

# Fragment a water cluster
system = af.io.read_xyz("water20.xyz")
partitioner = MolecularPartitioner(n_fragments=4, method="kmeans")
tree = partitioner.partition(system)

# Write input files for fragment-based calculations
write_gamess_fmo(tree.fragments, "water_fmo.inp", basis="aug-cc-pVDZ", method="MP2")
```

See the [Water Clusters Tutorial](../guides/water_clusters.md) for a complete
workflow.

### Energy Assembly

For MBE(2):

```python
E_total = sum(monomer_energies)
for i, j in dimer_pairs:
    E_total += dimer_energies[(i,j)] - monomer_energies[i] - monomer_energies[j]
```

For MBE(3):

```python
E_total = compute_mbe2()  # Start with MBE(2)
for i, j, k in trimer_triples:
    delta_3 = (trimer_energies[(i,j,k)] 
               - dimer_energies[(i,j)] - dimer_energies[(j,k)] - dimer_energies[(i,k)]
               + monomer_energies[i] + monomer_energies[j] + monomer_energies[k])
    E_total += delta_3
```

### Gradient Assembly

MBE gradients follow the same pattern:

$$
\frac{\partial E_{\text{MBE}}}{\partial R_\alpha} = \sum_i \frac{\partial E_i}{\partial R_\alpha} + \sum_{i<j} \frac{\partial \Delta E_{ij}}{\partial R_\alpha} + \cdots
$$

Each fragment gradient only affects coordinates within that fragment.

## Basis Set Superposition Error (BSSE)

### The Problem

When computing interaction energies, fragments can artificially lower their energy by using basis functions from neighboring fragments.

### Counterpoise Correction

The Boys-Bernardi counterpoise correction for dimer interaction:

$$
\Delta E_{ij}^{\text{CP}} = E_{ij}^{AB} - E_i^{AB} - E_j^{AB}
$$

where superscript $AB$ indicates calculation in the combined basis of fragments $A$ and $B$.

For MBE, apply counterpoise at each order with the full basis of all interacting fragments.

## Comparison with Other Fragment Methods

| Method | Overlap | Embedding | Accuracy | Scaling |
|--------|---------|-----------|----------|---------|
| MBE | No | Optional | Order-dependent | $O(N^n)$ |
| FMO | No | Electrostatic | High | $O(N^2)$ |
| SMF | Yes | None | Size-dependent | $O(N)$ |
| EE-MBE | No | Point charges | High | $O(N^2)$ |

## Applications

### Water Clusters

Ideal for MBE(2-3):
- Weak, non-covalent interactions
- Fast convergence
- Accurate with counterpoise

### Molecular Crystals

MBE for lattice energies:
- Periodic images become fragments
- Two-body captures most cohesive energy
- Three-body improves by ~5%

### Reaction Energies

Compute reaction energies as difference:

$$
\Delta E_{\text{rxn}} = E_{\text{products}}^{\text{MBE}} - E_{\text{reactants}}^{\text{MBE}}
$$

Error cancellation often improves accuracy.

## References

1. Dahlke, E. E., & Truhlar, D. G. (2007). Electrostatically embedded many-body expansion for large systems. *JCTC*, 3(1), 46-53.
2. Fedorov, D. G., & Kitaura, K. (2007). Pair interaction energy decomposition analysis. *JCC*, 28(1), 222-237.
3. Richard, R. M., & Herbert, J. M. (2012). A generalized many-body expansion and a unified view of fragment-based methods. *JCP*, 137(6), 064113.
4. Collins, M. A., & Bettens, R. P. (2015). Energy-based molecular fragmentation methods. *Chemical Reviews*, 115(12), 5607-5642.
5. Beran, G. J. (2016). Modeling polymorphic molecular crystals with electronic structure theory. *Chemical Reviews*, 116(9), 5567-5613.
