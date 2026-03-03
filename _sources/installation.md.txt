# Installation

## Requirements

- Python 3.10-3.12
- `pip` or `conda`

## Conda (recommended)

Create the runtime environment and install from source:

```bash
conda env create -f environment.yml
conda activate autofragment
python -m pip install .
```

For development, tests, and docs:

```bash
conda env create -f environment-dev.yml
conda activate autofragment-dev
python -m pip install -e ".[dev,docs]"
```

## Pip (PyPI)

```bash
python -m pip install --upgrade pip
python -m pip install autofragment
```

## Pip fallback (from source)

Use this path when conda is unavailable or you want a minimal local install:

```bash
git clone https://github.com/brycewestheimer/autofragment-public.git
cd autofragment-public
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Optional extras

```bash
python -m pip install "autofragment[balanced]"  # constrained k-means
python -m pip install "autofragment[bio]"       # mmCIF support (gemmi)
python -m pip install "autofragment[graph]"     # graph algorithms
python -m pip install "autofragment[matsci]"    # periodic/materials workflows
python -m pip install "autofragment[docs]"      # Sphinx + Breathe + RTD theme
python -m pip install "autofragment[all]"       # all runtime extras
```

## Verify installation

```bash
python -c "import autofragment; print(autofragment.__version__)"
```

## Coverage workflow

Install development dependencies and run coverage locally:

```bash
python -m pip install -e ".[dev]"
pytest tests -v --cov=autofragment --cov-report=xml --cov-report=term-missing
```

CI uploads `coverage.xml` to Codecov from GitHub Actions.
