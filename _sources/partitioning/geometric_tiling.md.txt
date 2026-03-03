# Geometric Tiling Partitioning

Geometric tiling partitioning assigns atoms to space-filling parallelohedra. This
is useful for regular, lattice-aligned fragmentations in periodic or finite
systems.

## Available tiling shapes

| Shape | Registry name | Notes |
| --- | --- | --- |
| Cube | `cube` | Axis-aligned cubic cells. |
| Hexagonal prism | `hex_prism` | Hexagonal cross-section, z-axis height. |
| Truncated octahedron | `truncated_octahedron` | BCC Wigner-Seitz cell. |
| Elongated dodecahedron | `elongated_dodecahedron` | Stretched rhombic dodecahedron. |
| Rhombic dodecahedron | `rhombic_dodecahedron` | FCC Wigner-Seitz cell. |

## CLI usage

Use the `geometric` subcommand to partition a system using tiling cells:

```bash
autofragment geometric \
  --input sample.xyz \
  --output sample.json \
  --tiling-shape cube \
  --n-fragments 8 \
  --tiling-scale 1.0
```

Supported CLI options:

- `--tiling-shape`: name from the registry table above
- `--n-fragments`: target number of fragments
- `--tiling-scale`: scale factor for the tiling lattice
- `--bounds-strategy`: `auto`, `lattice`, or `atoms`

## Python API

```python
from autofragment.partitioners.geometric import TilingPartitioner
from autofragment.core.types import Atom, ChemicalSystem

atoms = [Atom("H", [0.5, 0.5, 0.5]), Atom("H", [1.5, 1.5, 1.5])]
system = ChemicalSystem(atoms=atoms)

partitioner = TilingPartitioner(
    tiling_shape="cube",
    n_fragments=8,
    scale=1.0,
)
tree = partitioner.partition(system)
```

## Output metadata

Partitioning metadata records the tiling configuration:

```json
{
  "partitioning": {
    "algorithm": "tiling",
    "n_fragments": 8,
    "tiling_shape": "cube",
    "scale": 1.0,
    "bounds_strategy": "auto",
    "periodic": false
  }
}
```

Each fragment includes per-cell metadata:

```json
{
  "metadata": {
    "tiling_shape": "cube",
    "cell_index": [0, 1, 0],
    "scale": 1.0
  }
}
```
