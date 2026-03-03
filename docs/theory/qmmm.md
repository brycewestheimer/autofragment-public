# QM/MM and Embedding Theory

## Introduction

Hybrid Quantum Mechanics/Molecular Mechanics (QM/MM) methods partition a molecular system into regions treated with different levels of theory. The active site or region of interest is treated quantum mechanically, while the surrounding environment is treated with classical force fields.

This approach enables:
- Accurate treatment of chemically important regions
- Inclusion of environmental effects
- Computational tractability for large systems

## QM/MM Energy Expression

The total QM/MM energy is partitioned as:

$$
E_{\text{QM/MM}} = E_{\text{QM}} + E_{\text{MM}} + E_{\text{QM-MM}}
$$

where:
- $E_{\text{QM}}$: Quantum mechanical energy of the QM region
- $E_{\text{MM}}$: Molecular mechanics energy of the MM region
- $E_{\text{QM-MM}}$: Coupling between QM and MM regions

The coupling term $E_{\text{QM-MM}}$ depends on the **embedding scheme**.

## Embedding Schemes

### Mechanical Embedding

The simplest approach—only non-bonded classical interactions between regions:

$$
E_{\text{QM-MM}}^{\text{mech}} = \sum_{i \in \text{QM}} \sum_{j \in \text{MM}} \left[ \frac{q_i q_j}{r_{ij}} + E_{\text{LJ}}(r_{ij}) \right]
$$

where:
- $q_i$, $q_j$ are partial atomic charges
- $E_{\text{LJ}}$ is Lennard-Jones (van der Waals) interaction

**QM calculation**: Performed in vacuum (no MM influence on wavefunction)

**Advantages**:
- Simple implementation
- Fast QM calculations
- No special QM-MM interface needed

**Disadvantages**:
- QM region not polarized by environment
- May miss important electronic effects

### Electrostatic Embedding

MM point charges polarize the QM wavefunction:

$$
\hat{H}_{\text{QM}} = \hat{H}_0 + \sum_{j \in \text{MM}} \sum_k^{\text{elec}} \frac{-q_j}{|\mathbf{r}_k - \mathbf{R}_j|}
$$

where:
- $\hat{H}_0$ is the isolated QM Hamiltonian
- $\mathbf{r}_k$ is electron position
- $\mathbf{R}_j$ is MM atom position
- $q_j$ is MM point charge

The QM-MM electrostatic interaction energy:

$$
E_{\text{QM-MM}}^{\text{elec}} = \sum_{j \in \text{MM}} \sum_{A \in \text{QM}} \frac{Z_A \, q_j}{|\mathbf{R}_A - \mathbf{R}_j|} + \int \rho(\mathbf{r}) \, V_{\text{MM}}(\mathbf{r}) \, d\mathbf{r}
$$

where:
- $Z_A$ is nuclear charge of QM atom
- $\rho(\mathbf{r})$ is electron density
- $V_{\text{MM}}(\mathbf{r}) = \sum_j \frac{q_j}{|\mathbf{r} - \mathbf{R}_j|}$ is MM electrostatic potential

**Advantages**:
- QM responds to MM environment
- Captures polarization of active site
- Appropriate for charged or polar environments

**Disadvantages**:
- MM charges are fixed (no back-polarization)
- Charge overpolarization near boundary
- QM code must support external point charges

### Polarizable Embedding

Both regions mutually polarize each other:

$$
\boldsymbol{\mu}_j = \alpha_j \mathbf{E}(\mathbf{R}_j)
$$

where:
- $\boldsymbol{\mu}_j$ is the induced dipole on MM atom $j$
- $\alpha_j$ is the atomic polarizability
- $\mathbf{E}(\mathbf{R}_j)$ is the total electric field at $j$

The electric field includes contributions from:
- QM nuclear charges
- QM electron density
- Other MM charges and induced dipoles

**Self-consistent iteration**:
1. Compute QM density with current MM dipoles
2. Update MM induced dipoles from QM field
3. Repeat until convergence

**Advantages**:
- Most accurate embedding
- Captures mutual polarization
- Appropriate for highly polarizable environments

**Disadvantages**:
- Requires iterative SCF
- More expensive
- Force field must include polarizabilities

### Comparison of Embedding Schemes

| Feature | Mechanical | Electrostatic | Polarizable |
|---------|------------|---------------|-------------|
| QM cost | Lowest | Medium | Highest |
| MM effect on QM | None | Charges | Charges + dipoles |
| QM effect on MM | None | None | Induced dipoles |
| Accuracy | Low | Medium | High |
| Implementation | Simple | Moderate | Complex |

## Boundary Treatments

When covalent bonds cross the QM/MM boundary, special treatment is required.

### Link Atoms

Replace the MM atom at the boundary with a hydrogen-like atom:

$$
\mathbf{R}_{\text{link}} = \mathbf{R}_{\text{QM}} + g \cdot (\mathbf{R}_{\text{MM}} - \mathbf{R}_{\text{QM}})
$$

where:
- $\mathbf{R}_{\text{QM}}$ is the QM boundary atom position
- $\mathbf{R}_{\text{MM}}$ is the MM boundary atom position  
- $g$ is the **g-factor** scaling the QM-MM distance to QM-H distance

#### G-Factor Calculation

$$
g = \frac{r_{\text{QM-H}}}{r_{\text{QM-MM}}}
$$

Typical values:

| Bond Type | $r_{\text{QM-MM}}$ (Å) | $r_{\text{QM-H}}$ (Å) | g-factor |
|-----------|----------------------|---------------------|----------|
| C-C | 1.54 | 1.09 | 0.708 |
| C-N | 1.47 | 1.09 | 0.741 |
| C-O | 1.43 | 1.09 | 0.762 |

#### Link Atom Force Distribution

Forces on link atoms must be redistributed:

$$
\mathbf{F}_{\text{QM}} = \mathbf{F}_{\text{link}} \cdot g + \mathbf{F}_{\text{QM}}^{\text{direct}}
$$

$$
\mathbf{F}_{\text{MM}} = \mathbf{F}_{\text{link}} \cdot (1 - g) + \mathbf{F}_{\text{MM}}^{\text{direct}}
$$

### Charge Redistribution

Avoid overpolarization by MM charges close to QM region:

**Charge shifting**: Zero charges within cutoff, add to neighbors

**Charge scaling**: Gradually scale charges near boundary

**Gaussian blur**: Replace point charges with Gaussian distributions:

$$
q_j(\mathbf{r}) = q_j \left( \frac{\alpha}{\pi} \right)^{3/2} e^{-\alpha|\mathbf{r} - \mathbf{R}_j|^2}
$$

### Frontier Bonds

Constraint treatment of the QM-MM bond:

- **Distance constraint**: Fix QM-MM distance
- **Harmonic restraint**: Soft restraint to equilibrium distance
- **Bond parameterization**: Include in MM force field

## ONIOM Method

ONIOM (Our own N-layered Integrated molecular Orbital and molecular mechanics) is an extrapolation scheme combining multiple levels of theory.

### Two-Layer ONIOM

$$
E_{\text{ONIOM}} = E_{\text{real}}^{\text{low}} + E_{\text{model}}^{\text{high}} - E_{\text{model}}^{\text{low}}
$$

where:
- **real**: Full system
- **model**: Inner (high-level) region only
- **high**: High-level method (e.g., DFT)
- **low**: Low-level method (e.g., MM or semi-empirical)

This extrapolates the high-level energy to the full system by correcting for environmental effects at low level.

### Three-Layer ONIOM

For three regions (inner, middle, outer):

$$
\begin{align}
E_{\text{ONIOM}} &= E_{\text{real}}^{\text{low}} \\
&\quad + E_{\text{middle}}^{\text{medium}} - E_{\text{middle}}^{\text{low}} \\
&\quad + E_{\text{inner}}^{\text{high}} - E_{\text{inner}}^{\text{medium}}
\end{align}
$$

### ONIOM Gradients

Gradients follow same pattern with chain rule for link atoms:

$$
\nabla E_{\text{ONIOM}} = \nabla E_{\text{real}}^{\text{low}} + J^T \left( \nabla E_{\text{model}}^{\text{high}} - \nabla E_{\text{model}}^{\text{low}} \right)
$$

where $J$ is the Jacobian relating model coordinates to real coordinates.

## Effective Fragment Potential (EFP)

EFP represents MM fragments with multipoles extracted from QM calculations.

### EFP Energy Decomposition

$$
E_{\text{EFP}} = E_{\text{Coulomb}} + E_{\text{polarization}} + E_{\text{dispersion}} + E_{\text{exchange-repulsion}} + E_{\text{charge-transfer}}
$$

Each term is computed from:
- **Coulomb**: Distributed multipoles (through octupoles)
- **Polarization**: Distributed polarizabilities
- **Dispersion**: C6 coefficients
- **Exchange-repulsion**: Exponential overlap
- **Charge-transfer**: Optional donor-acceptor interaction

### Advantages over Point-Charge MM

- Derived from QM (no empirical fitting)
- Includes higher multipoles (better electrostatics)
- Natural polarization treatment
- Transferable between systems

## Implementation in AutoFragment

### Basic QM/MM Setup

```python
from autofragment.partitioners import QMMMPartitioner
from autofragment.multilevel import EmbeddingType

partitioner = QMMMPartitioner(
    qm_selection=selection,
    buffer_radius=5.0,
    link_scheme="hydrogen",
    embedding=EmbeddingType.ELECTROSTATIC
)

result = partitioner.partition(molecules)

# Access link atoms
for la in result.link_atoms:
    print(f"Link H at position {la.position}")

# Generate point charges
charges = result.generate_mm_charges()
```

### ONIOM Setup

```python
from autofragment.multilevel import ONIOMScheme

scheme = ONIOMScheme.from_string("ONIOM(B3LYP/6-31G*:UFF)")
scheme.set_layer_atoms("high", high_indices)
scheme.set_layer_atoms("low", low_indices)

# Generate input
gaussian_input = scheme.to_gaussian_input()
```

## Best Practices

### QM Region Selection

- Include all reactive atoms/bonds
- Complete any rings or conjugated systems
- Add 1-2 residues beyond reaction center (proteins)
- Consider 5-10 Å for charged species

### Embedding Scheme Choice

| System | Recommended Embedding |
|--------|----------------------|
| Enzyme active site | Electrostatic |
| Metal in protein | Polarizable |
| Small molecule in solution | EFP |
| Material surface | Mechanical (with correction) |

### Link Atom Placement

- Place at single, non-polar bonds (C-C ideal)
- Avoid cutting near charges or polar groups
- Match link atom to attached MM carbon

## References

1. Warshel, A., & Levitt, M. (1976). Theoretical studies of enzymic reactions. *JMB*, 103(2), 227-249.
2. Dapprich, S., et al. (1999). A new ONIOM implementation in Gaussian. *Journal of Molecular Structure*, 461, 1-21.
3. Senn, H. M., & Thiel, W. (2009). QM/MM methods for biomolecular systems. *Angewandte Chemie International Edition*, 48(7), 1198-1229.
4. Gordon, M. S., et al. (2012). Accurate methods for large molecular systems. *JPC B*, 113(29), 9646-9660.
5. Lin, H., & Truhlar, D. G. (2007). QM/MM: what have we learned, where are we, and where do we go from here? *Theoretical Chemistry Accounts*, 117(2), 185-199.
