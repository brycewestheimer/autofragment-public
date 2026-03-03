# Contributing to AutoFragment

Thank you for your interest in contributing to AutoFragment! We welcome contributions from the community to make this tool better.

## Development Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/brycewestheimer/autofragment-public.git
    cd autofragment-public
    ```

2.  **Create a development environment**
    We recommend using conda:
    ```bash
    conda env create -f environment-dev.yml
    conda activate autofragment-dev
    pip install -e ".[dev,docs]"
    ```

3.  **Run tests**
    Ensure everything is working correctly:
    ```bash
    pytest
    ```

## Code Style

-   We use [Ruff](https://beta.ruff.rs/docs/) for linting and formatting.
-   All new code must include type hints and docstrings (NumPy style).
-   Please ensure your editor is configured to run formatting on save, or run:
    ```bash
    ruff check . --fix
    ruff format .
    ```

## Pull Request Process

1.  Create a new branch for your feature or fix: `git checkout -b feature/my-new-feature`.
2.  Add tests that cover your new functionality.
3.  Implement your changes.
4.  Run the full test suite to ensure no regressions.
5.  Update documentation if necessary (including `CHANGELOG.md`).
6.  Submit a Pull Request.

## Documentation

Documentation is built with Sphinx. To build locally:

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` to view.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub.
