# Performance Profiling

This document summarizes profiling results and expected hot paths for large systems.

## Profiling Script

Run the profiling script against a large PDB file:

```bash
python benchmarks/profile_partitioning.py
```

## Expected Hot Paths

1. **Distance calculations** (pairwise distance checks during bond inference).
2. **Neighbor finding** (graph traversal for connectivity).
3. **Rule matching** (fragmentation rule evaluation).
4. **Bond detection** (covalent radius checks).
5. **Graph partitioning** (min-cut or community detection algorithms).

## Baseline Notes

- Large system profiling should focus on PDBs in the 10k–100k atom range.
- Store your baseline output in version control if the dataset is stable.
- Use `line_profiler` and `memory_profiler` for deeper inspection.

## Suggested Commands

```bash
pip install line_profiler memory_profiler
kernprof -l -v benchmarks/profile_partitioning.py
python -m memory_profiler benchmarks/profile_partitioning.py
```
