# Command Line

## Top-level shortcut (single file)

```bash
autofragment --input water64.xyz --output partitioned.json --n-fragments 4
```

## Fragment a single XYZ

```bash
autofragment single --input water64.xyz --output partitioned.json --n-fragments 4
```

## Batch mode (consistent labeling)

```bash
autofragment batch --reference ref.xyz --input-dir ./trajectory/ --output-dir ./partitioned/ --n-fragments 4
```

## Biological systems (mmCIF)

```bash
autofragment bio --input protein.cif --output partitioned.json
```

## Geometric tiling mode

```bash
autofragment geometric --input water64.xyz --output tiled.json --tiling-shape cube --n-fragments 8
```

## Tiered (hierarchical) partitioning

```bash
# 2-tier: 4 primary x 4 secondary
autofragment single --input water64.xyz --output tiered.json \
    --tiers 2 --n-primary 4 --n-secondary 4 --method kmeans

# 3-tier: 2 primary x 2 secondary x 2 tertiary
autofragment single --input water64.xyz --output tiered.json \
    --tiers 3 --n-primary 2 --n-secondary 2 --n-tertiary 2

# With seeding strategy
autofragment single --input water64.xyz --output tiered.json \
    --tiers 2 --n-primary 4 --n-secondary 4 --init-strategy pca

# Batch mode with tiered partitioning
autofragment batch --reference ref.xyz --input-dir ./trajectory/ --output-dir ./partitioned/ \
    --tiers 2 --n-primary 4 --n-secondary 4
```

## Show version and optional features

```bash
autofragment info
autofragment --version
```
