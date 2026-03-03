# Development

## Setup

```bash
git clone https://github.com/brycewestheimer/autofragment-public.git
cd autofragment-public
conda env create -f environment-dev.yml
conda activate autofragment-dev
python -m pip install -e ".[dev,docs]"
```

## Quality checks

```bash
ruff check src tests
mypy src/autofragment
pytest tests -v
```

## Coverage

```bash
pytest tests -v --cov=autofragment --cov-report=xml --cov-report=term-missing
```

## Documentation

Documentation uses Sphinx with MyST, Breathe, and the Read the Docs theme.

```bash
sphinx-build -b html docs docs/_build/html
```

## Packaging

```bash
python -m pip install build twine
python -m build
twine check dist/*
```
