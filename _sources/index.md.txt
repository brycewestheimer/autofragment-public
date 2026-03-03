# autofragment

Molecular fragmentation for computational chemistry.

`autofragment` partitions molecular systems (water clusters, proteins, etc.) into fragments suitable for many-body expansion calculations. Supports both **flat** partitioning and **tiered hierarchical** fragmentation (2-tier and 3-tier).

```{toctree}
:maxdepth: 2
:caption: Getting Started

guides/quickstart
examples
installation
cli

```

```{toctree}
:maxdepth: 2
:caption: Tutorials

guides/water_clusters
guides/proteins
guides/materials
guides/custom_rules
guides/output_formats
partitioning/geometric_tiling

```

```{toctree}
:maxdepth: 2
:caption: User Guide

python_api
io_formats
rules
guides/algorithms
biological
materials
multilevel
core_concepts
migration
output

```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/clustering
theory/graph_partitioning
theory/scoring
theory/mbe
theory/qmmm

```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index

```

```{toctree}
:maxdepth: 1
:caption: Development

development
dev/performance
```
