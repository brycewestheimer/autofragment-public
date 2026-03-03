# Documentation (Sphinx)

This project uses **Sphinx** to build documentation from the files in `docs/`.

## Build locally

From the repo root:

```bash
pip install -e ".[docs]"
make -C docs html
```

Open the generated site:

- `docs/_build/html/index.html`

## Clean

```bash
make -C docs clean
```
