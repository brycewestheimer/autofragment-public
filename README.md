# autofragment

[![CI](https://github.com/brycewestheimer/autofragment-public/actions/workflows/test.yml/badge.svg)](https://github.com/brycewestheimer/autofragment-public/actions/workflows/test.yml)
[![Docs](https://github.com/brycewestheimer/autofragment-public/actions/workflows/docs.yml/badge.svg)](https://github.com/brycewestheimer/autofragment-public/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/brycewestheimer/autofragment-public/branch/main/graph/badge.svg)](https://codecov.io/gh/brycewestheimer/autofragment-public)

`autofragment` is a production-ready Python library for chemistry-aware molecular fragmentation in computational chemistry workflows.

Version: **1.0.0**

## Features

- Molecular, biological, materials, and QM/MM partitioning workflows.
- Multiple partitioning strategies: geometric, graph-based, spectral, and clustering approaches.
- Tiered hierarchical fragmentation: 2-tier and 3-tier recursive partitioning with configurable primary/secondary/tertiary fragment counts.
- K-means seeding strategies (halfplane, PCA, axis, radial) with per-tier overrides.
- Rule-aware fragmentation constraints for chemistry and domain-specific behavior.
- Input support for common structural formats; output writers for major QC ecosystems.
- Python API and CLI for both single-system and batch fragmentation workflows.

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate autofragment
python -m pip install .
```

### Pip (published package)

```bash
python -m pip install --upgrade pip
python -m pip install autofragment
```

### Pip fallback from source

```bash
git clone https://github.com/brycewestheimer/autofragment-public.git
cd autofragment-public
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

### Optional extras

```bash
python -m pip install "autofragment[balanced]"
python -m pip install "autofragment[bio]"
python -m pip install "autofragment[graph]"
python -m pip install "autofragment[matsci]"
python -m pip install "autofragment[all]"
```

## Quick start

### Python API

```python
import autofragment as af
from autofragment.partitioners import MolecularPartitioner

system = af.io.read_xyz("examples/data/water_64.xyz")
partitioner = MolecularPartitioner(n_fragments=8, method="kmeans")
result = partitioner.partition(system)
print(len(result.fragments))
```

### Tiered (hierarchical) mode

```python
partitioner = MolecularPartitioner(
    tiers=2, n_primary=4, n_secondary=4, method="kmeans"
)
result = partitioner.partition(system)
print(result.n_primary)  # 4 primary fragments, each with 4 sub-fragments
```

### CLI

```bash
autofragment single --input examples/data/water_64.xyz --output partitioned.json --n-fragments 8

# Tiered mode
autofragment single --input examples/data/water_64.xyz --output partitioned.json \
    --tiers 2 --n-primary 4 --n-secondary 4 --method kmeans
```

## Documentation

- Source docs: `docs/`
- Entry page: `docs/index.md`
- Build locally:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Documentation tooling is configured for **Sphinx + Breathe + Read the Docs theme**.

## Testing and coverage

```bash
python -m pip install -e ".[dev]"
pytest tests -v
pytest tests -v --cov=autofragment --cov-report=xml --cov-report=term-missing
```

CI uploads coverage reports to Codecov.

## CI/CD

GitHub Actions workflows are configured for:

- Linting and type checks.
- Multi-OS and multi-version test matrix.
- Coverage reporting.
- Package build verification.
- Conda installation validation.
- Pip fallback installation validation.
- Documentation build and deployment.
- Release publishing workflow.

## Citation

Citation metadata is provided in `CITATION.cff`.

No manuscript is currently published; manuscript is in preparation.

## Contributing

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

## License

BSD-3-Clause. See `LICENSE`.

## Author

Bryce M. Westheimer  
GitHub: <https://github.com/brycewestheimer>
