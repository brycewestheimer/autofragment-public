.PHONY: help install dev test docs build clean

help:
	@echo "Targets:"
	@echo "  install  - Install package (pip)"
	@echo "  dev      - Editable install with dev deps"
	@echo "  test     - Run pytest"
	@echo "  docs     - Build Sphinx docs"
	@echo "  build    - Build sdist+wheel into dist/"
	@echo "  clean    - Remove build/test artifacts"

install:
	python -m pip install -U pip
	python -m pip install .

dev:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

docs:
	python -m pip install -U pip
	python -m pip install -e ".[docs]"
	$(MAKE) -C docs html

build:
	python -m pip install -U pip
	python -m pip install build
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	rm -rf docs/_build
