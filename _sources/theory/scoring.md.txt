# Fragmentation Scoring Functions

## Overview

Scoring functions quantify the quality of a molecular fragmentation to guide optimization algorithms. A good fragmentation should:

1. Minimize breaking of strong chemical bonds
2. Produce fragments of reasonable, uniform size
3. Minimize interfragment interactions
4. Preserve chemical integrity (functional groups, rings)
5. Balance computational cost across fragments

## Total Score Formulation

The total score is a weighted combination of component scores:

$$
S_{\text{total}} = \sum_{i} w_i \cdot S_i
$$

where:
- $w_i$ are tunable weights for each component
- $S_i$ are individual score components
- Higher scores indicate better fragmentations

Alternative formulation as minimization (penalty-based):

$$
P_{\text{total}} = \sum_{i} w_i \cdot P_i
$$

where $P_i$ are penalty terms and lower values are better.

## Component Scores

### Bond Breaking Penalty ($S_{\text{bond}}$)

The most critical component—penalizes breaking covalent bonds:

$$
S_{\text{bond}} = -\sum_{b \in \text{broken}} E_{\text{bond}}(b) \cdot f_{\text{type}}(b)
$$

where:
- $E_{\text{bond}}(b)$ is the estimated bond energy
- $f_{\text{type}}(b)$ is a multiplier based on bond type

#### Bond Type Multipliers

| Bond Type | $f_{\text{type}}$ | Rationale |
|-----------|-------------------|-----------|
| Single (rotatable) | 1.0 | Preferred break points |
| Single (non-rotatable) | 1.5 | Somewhat penalized |
| Aromatic | 10.0 | Strongly discouraged |
| Double | 50.0 | Almost forbidden |
| Triple | 100.0 | Forbidden |

#### Bond Energy Estimation

Approximate bond energies (kcal/mol):

$$
E_{\text{bond}}(X-Y) \approx \sqrt{E(X-X) \cdot E(Y-Y)}
$$

| Bond | Energy (kcal/mol) |
|------|-------------------|
| C-C | 83 |
| C-H | 99 |
| C-N | 73 |
| C-O | 86 |
| C=O | 180 |
| C=C | 146 |
| C≡C | 200 |

### Size Variance ($S_{\text{size}}$)

Encourages fragments of uniform size to balance computational load:

$$
S_{\text{size}} = -\alpha \cdot \frac{\sigma(n_i)}{\bar{n}}
$$

where:
- $n_i$ is the number of atoms in fragment $i$
- $\bar{n} = \frac{1}{k}\sum_i n_i$ is the mean fragment size
- $\sigma(n_i) = \sqrt{\frac{1}{k}\sum_i (n_i - \bar{n})^2}$ is the standard deviation
- $\alpha$ is a scaling parameter

**Alternative**: Use coefficient of variation as penalty:

$$
\text{CV} = \frac{\sigma(n_i)}{\bar{n}}
$$

Normalized score:

$$
S_{\text{size}} = 1 - \text{CV}
$$

### Size Range Penalty

Penalize fragments outside acceptable size range:

$$
P_{\text{range}} = \sum_i \max(0, n_i - n_{\max}) + \sum_i \max(0, n_{\min} - n_i)
$$

where $n_{\min}$ and $n_{\max}$ are target fragment size bounds.

### Interface Score ($S_{\text{interface}}$)

Minimizes interaction energy across fragment boundaries:

$$
S_{\text{interface}} = -\sum_{(F_i, F_j)} E_{\text{int}}(F_i, F_j)
$$

#### Approximations for Interface Energy

**Exposed surface area**:

$$
E_{\text{int}} \approx \gamma \cdot A_{\text{exposed}}
$$

where $\gamma$ is surface tension and $A_{\text{exposed}}$ is solvent-accessible surface area at cut points.

**Electrostatic interaction**:

$$
E_{\text{int}} \approx \sum_{a \in F_i} \sum_{b \in F_j} \frac{q_a q_b}{r_{ab}}
$$

**Number of cut bonds** (simplified):

$$
E_{\text{int}} \approx N_{\text{cut}}
$$

### Chemical Integrity ($S_{\text{chem}}$)

Rewards preservation of chemically meaningful units:

$$
S_{\text{chem}} = \sum_f I_{\text{valid}}(f)
$$

where $I_{\text{valid}}(f)$ is an indicator for chemically sensible fragments.

#### Validity Criteria

1. **No broken rings**: 
   $$
   I_{\text{ring}}(f) = \begin{cases}
   0 & \text{if ring is split} \\
   1 & \text{otherwise}
   \end{cases}
   $$

2. **Complete functional groups**:
   $$
   I_{\text{func}}(f) = \begin{cases}
   0 & \text{if functional group is split} \\
   1 & \text{otherwise}
   \end{cases}
   $$

3. **Proper valence**:
   $$
   I_{\text{valence}}(f) = \begin{cases}
   1 & \text{if all atoms have valid valence (after capping)} \\
   0 & \text{otherwise}
   \end{cases}
   $$

### Computational Cost ($S_{\text{comp}}$)

Estimates total computational cost of the fragmentation:

$$
S_{\text{comp}} = -\sum_i C(n_i)
$$

where $C(n)$ is the cost function for QM calculations.

#### Scaling Estimates

| Method | Scaling | $C(n)$ |
|--------|---------|--------|
| HF | $O(n^4)$ | $n^4$ |
| DFT | $O(n^3)$ | $n^3$ |
| MP2 | $O(n^5)$ | $n^5$ |
| CCSD | $O(n^6)$ | $n^6$ |
| CCSD(T) | $O(n^7)$ | $n^7$ |

For many-body expansion, include pair interaction costs:

$$
S_{\text{comp}}^{\text{MBE}} = -\left[ \sum_i C(n_i) + \frac{1}{2}\sum_{i \neq j} C(n_i + n_j) \right]
$$

### Distance from Target Size ($S_{\text{target}}$)

Encourages fragments close to specified target size:

$$
S_{\text{target}} = -\sum_i \left| n_i - n_{\text{target}} \right|^p
$$

Common choices:
- $p = 1$: Linear penalty (MAE)
- $p = 2$: Quadratic penalty (MSE), more severe for large deviations

### Number of Fragments ($S_{\text{count}}$)

Penalize excessive fragmentation:

$$
S_{\text{count}} = -|k - k_{\text{target}}|
$$

where $k$ is the number of fragments.

## Weight Selection

### Default Weights

Balanced for general molecular fragmentation:

| Component | Weight | Rationale |
|-----------|--------|-----------|
| $w_{\text{bond}}$ | 1.0 | Primary consideration |
| $w_{\text{size}}$ | 0.3 | Moderate importance |
| $w_{\text{interface}}$ | 0.2 | Secondary |
| $w_{\text{chem}}$ | 0.5 | Important for validity |
| $w_{\text{comp}}$ | 0.1 | Use when balancing load |

### Application-Specific Weights

**FMO calculations** (prioritize size balance):
```python
weights = {
    "bond": 1.0,
    "size": 0.8,  # Higher for balanced fragments
    "chem": 1.0,  # Critical for FMO
    "interface": 0.1,
}
```

**Large protein fragmentation** (prioritize computational cost):
```python
weights = {
    "bond": 1.0,
    "size": 0.3,
    "comp": 0.5,  # Consider cost
    "chem": 0.5,
}
```

**Water clusters** (minimize interactions):
```python
weights = {
    "bond": 0.5,  # Weak H-bonds
    "interface": 1.0,  # Minimize water-water
    "size": 0.1,  # Less important
}
```

## Multi-Objective Optimization

### Pareto Optimality

Multiple objectives can conflict. A fragmentation is **Pareto optimal** if no objective can be improved without worsening another.

The **Pareto frontier** contains all Pareto-optimal fragmentations:

$$
\mathcal{P} = \{f : \nexists f' \text{ such that } S_i(f') \geq S_i(f) \forall i \text{ and } S_j(f') > S_j(f) \text{ for some } j\}
$$

### Scalarization

Convert to single objective via weighted sum:

$$
S_{\text{total}}(\mathbf{w}) = \sum_i w_i S_i
$$

Different weight vectors explore different points on Pareto frontier.

### ε-Constraint Method

Optimize one objective subject to constraints on others:

$$
\maximize \quad S_1 \\
\text{subject to} \quad S_i \geq \epsilon_i, \quad i = 2, \ldots, k
$$

## Trade-offs and Considerations

### Bond Breaking vs. Size Balance

- Fewer cuts → larger fragments → worse computational scaling
- More cuts → smaller fragments → more interface energy

Optimal balance depends on:
- Target QM method (scaling exponent)
- System type (proteins need larger fragments than water)
- Accuracy requirements

### Chemical Integrity vs. Flexibility

Strict preservation of all functional groups may produce:
- Fragments that are too large
- Poor size balance
- Suboptimal partitioning

Consider relaxing constraints for:
- Long-chain functional groups
- Repeated patterns (polymers)

### Computational Cost vs. Accuracy

**Many-body expansion trade-off**:
- Fewer, larger fragments → fewer pairs → faster → less accurate
- More, smaller fragments → more pairs → slower → more accurate

Target fragment size should be chosen based on desired accuracy.

## Score Normalization

### Z-Score Normalization

For comparing across systems:

$$
S_i^{\text{norm}} = \frac{S_i - \mu_i}{\sigma_i}
$$

where $\mu_i$ and $\sigma_i$ are mean and std from reference fragmentations.

### Min-Max Scaling

$$
S_i^{\text{scaled}} = \frac{S_i - S_i^{\min}}{S_i^{\max} - S_i^{\min}}
$$

Bounds scores to $[0, 1]$ range.

## Implementation Notes

```python
from autofragment.scoring import FragmentationScorer

scorer = FragmentationScorer(
    weights={
        "bond": 1.0,
        "size": 0.3,
        "interface": 0.2,
        "chem": 0.5,
    }
)

# Score a fragmentation
fragments = partitioner.partition(system)
score = scorer.score(fragments)

# Get component breakdown
breakdown = scorer.score_breakdown(fragments)
for component, value in breakdown.items():
    print(f"{component}: {value:.3f}")
```

## References

1. Fedorov, D. G., & Kitaura, K. (2007). Extending the power of quantum chemistry to large systems with the fragment molecular orbital method. *JPC A*, 111(30), 6904-6914.
2. Gordon, M. S., et al. (2012). Fragmentation methods: A route to accurate calculations on large systems. *Chemical Reviews*, 112(1), 632-672.
3. Collins, M. A., & Bettens, R. P. (2015). Energy-based molecular fragmentation methods. *Chemical Reviews*, 115(12), 5607-5642.
