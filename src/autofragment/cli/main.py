# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line interface for autofragment.

Provides the main entry point and subcommands for:
- single: Partition a single XYZ file
- batch: Partition multiple files with consistent labeling
- bio: Partition biological systems from mmCIF
- info: Show version and feature information
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the autofragment CLI."""
    parser = argparse.ArgumentParser(
        prog="autofragment",
        description="Molecular fragmentation for computational chemistry",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    # Add single-command arguments to top-level parser so that
    # `autofragment --input file.xyz` works without specifying `single`.
    _add_single_args(parser, required_input=False)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single file command
    _add_single_parser(subparsers)

    # Batch command
    _add_batch_parser(subparsers)

    # Bio command
    _add_bio_parser(subparsers)

    # Geometric tiling command
    _add_geometric_parser(subparsers)

    # Info command
    _add_info_parser(subparsers)

    args = parser.parse_args(argv)

    if args.version:
        from autofragment._version import __version__

        print(f"autofragment {__version__}")
        return 0

    if args.command is None:
        if getattr(args, "input", None):
            return _cmd_single(args)
        parser.print_help()
        return 0

    return args.func(args)


def _add_single_args(parser, *, required_input: bool = True) -> None:
    """Add single-file partitioning arguments to *parser*.

    Shared between the top-level parser (where ``--input`` is optional) and
    the ``single`` subcommand (where it is required).
    """
    parser.add_argument("--input", "-i", required=required_input, help="Input XYZ file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: input.json)")

    frag_group = parser.add_argument_group("fragmentation options")
    frag_group.add_argument(
        "--n-fragments", type=int, default=4, help="Number of fragments (default: 4)"
    )
    frag_group.add_argument(
        "--atoms-per-molecule",
        type=int,
        default=None,
        help="Atoms per molecule (default: auto-detect, None=single molecule)",
    )
    frag_group.add_argument(
        "--no-validate-water",
        action="store_true",
        help="Disable water structure validation",
    )

    # Hierarchy options
    hier_group = parser.add_argument_group("hierarchy options")
    hier_group.add_argument(
        "--tiers", type=int, choices=[2, 3], default=None,
        help="Number of hierarchy tiers (default: None = flat mode)"
    )
    hier_group.add_argument(
        "--n-primary", type=int, default=None,
        help="Number of primary fragments (tiered mode)"
    )
    hier_group.add_argument(
        "--n-secondary", type=int, default=None,
        help="Number of secondary fragments per primary (tiered mode)"
    )
    hier_group.add_argument(
        "--n-tertiary", type=int, default=None,
        help="Number of tertiary fragments per secondary (3-tier mode)"
    )

    cluster_group = parser.add_argument_group("clustering options")
    cluster_group.add_argument(
        "--method",
        choices=[
            "kmeans",
            "agglomerative",
            "spectral",
            "kmeans_constrained",
            "gmm",
            "birch",
            "geom_planes",
        ],
        default="kmeans",
        help="Clustering method (default: kmeans)",
    )
    cluster_group.add_argument(
        "--random-state", type=int, default=42, help="Random seed for clustering (default: 42)"
    )

    # Seeding options
    seed_group = parser.add_argument_group("seeding options")
    seed_group.add_argument(
        "--init-strategy",
        choices=["halfplane", "pca", "axis", "radial"],
        help="K-means seeding strategy for all tiers (default: k-means++)"
    )
    seed_group.add_argument(
        "--init-strategy-primary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for primary (tier-1) clustering"
    )
    seed_group.add_argument(
        "--init-strategy-secondary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for secondary (tier-2) clustering"
    )
    seed_group.add_argument(
        "--init-strategy-tertiary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for tertiary (tier-3) clustering"
    )
    seed_group.add_argument(
        "--init-axis",
        choices=["x", "y", "z"],
        help="Cartesian axis for 'axis' seeding strategy"
    )

    parser.add_argument(
        "--xyz-units",
        choices=["angstrom", "bohr"],
        default="angstrom",
        help="Units of XYZ coordinates (default: angstrom)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")


def _add_single_parser(subparsers) -> None:
    """Add the 'single' subcommand."""
    parser = subparsers.add_parser(
        "single",
        help="Partition a single XYZ file",
        description="Partition a water cluster XYZ file into fragments.",
    )
    _add_single_args(parser, required_input=True)
    parser.set_defaults(func=_cmd_single)


def _add_batch_parser(subparsers) -> None:
    """Add the 'batch' subcommand."""
    parser = subparsers.add_parser(
        "batch",
        help="Partition multiple files with consistent labeling",
        description="Partition multiple XYZ files using a reference for consistent fragment labels.",
    )

    # Required arguments
    parser.add_argument(
        "--reference", required=True, help="Reference XYZ file (establishes labeling)"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files")

    # Input options (mutually exclusive-ish)
    input_group = parser.add_argument_group("input options")
    input_group.add_argument("--input-dir", help="Directory containing XYZ files")
    input_group.add_argument("--inputs", nargs="*", help="Specific XYZ files to process")
    input_group.add_argument(
        "--pattern", default="*.xyz", help="Glob pattern for input-dir (default: *.xyz)"
    )

    # Fragmentation options
    frag_group = parser.add_argument_group("fragmentation options")
    frag_group.add_argument(
        "--n-fragments", type=int, default=4, help="Number of fragments (default: 4)"
    )
    frag_group.add_argument(
        "--atoms-per-molecule",
        type=int,
        default=None,
        help="Atoms per molecule (default: auto-detect, None=single molecule)",
    )
    frag_group.add_argument(
        "--no-validate-water",
        action="store_true",
        help="Disable water structure validation",
    )

    # Hierarchy options
    hier_group = parser.add_argument_group("hierarchy options")
    hier_group.add_argument(
        "--tiers", type=int, choices=[2, 3], default=None,
        help="Number of hierarchy tiers (default: None = flat mode)"
    )
    hier_group.add_argument(
        "--n-primary", type=int, default=None,
        help="Number of primary fragments (tiered mode)"
    )
    hier_group.add_argument(
        "--n-secondary", type=int, default=None,
        help="Number of secondary fragments per primary (tiered mode)"
    )
    hier_group.add_argument(
        "--n-tertiary", type=int, default=None,
        help="Number of tertiary fragments per secondary (3-tier mode)"
    )

    # Clustering options
    cluster_group = parser.add_argument_group("clustering options")
    cluster_group.add_argument(
        "--method",
        choices=[
            "kmeans",
            "agglomerative",
            "spectral",
            "kmeans_constrained",
            "gmm",
            "birch",
            "geom_planes",
        ],
        default="kmeans",
        help="Clustering method (default: kmeans)",
    )
    cluster_group.add_argument(
        "--random-state", type=int, default=42, help="Random seed for clustering (default: 42)"
    )

    # Seeding options
    seed_group = parser.add_argument_group("seeding options")
    seed_group.add_argument(
        "--init-strategy",
        choices=["halfplane", "pca", "axis", "radial"],
        help="K-means seeding strategy for all tiers (default: k-means++)"
    )
    seed_group.add_argument(
        "--init-strategy-primary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for primary (tier-1) clustering"
    )
    seed_group.add_argument(
        "--init-strategy-secondary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for secondary (tier-2) clustering"
    )
    seed_group.add_argument(
        "--init-strategy-tertiary",
        choices=["halfplane", "pca", "axis", "radial"],
        help="Override seeding strategy for tertiary (tier-3) clustering"
    )
    seed_group.add_argument(
        "--init-axis",
        choices=["x", "y", "z"],
        help="Cartesian axis for 'axis' seeding strategy"
    )

    # Matching options
    match_group = parser.add_argument_group("matching options")
    match_group.add_argument(
        "--metric",
        choices=["centroid", "rmsd", "hybrid"],
        default="rmsd",
        help="Distance metric for matching (default: rmsd)",
    )
    match_group.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight for centroid in hybrid metric (default: 0.5)",
    )
    match_group.add_argument(
        "--align", action="store_true", help="Apply Kabsch alignment before matching"
    )
    match_group.add_argument(
        "--force", action="store_true", help="Proceed even if assignment quality is poor"
    )

    # Other options
    parser.add_argument(
        "--xyz-units",
        choices=["angstrom", "bohr"],
        default="angstrom",
        help="Units of XYZ coordinates (default: angstrom)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    parser.set_defaults(func=_cmd_batch)


def _add_bio_parser(subparsers) -> None:
    """Add the 'bio' subcommand."""
    parser = subparsers.add_parser(
        "bio",
        help="Partition biological systems from mmCIF",
        description="Partition biological structures (proteins, waters, ligands) from mmCIF files.",
    )

    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Input mmCIF file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: input.json)")

    # Bio options
    bio_group = parser.add_argument_group("biological options")
    bio_group.add_argument(
        "--water-clusters", type=int, help="Number of water clusters (auto if not set)"
    )
    bio_group.add_argument(
        "--water-cluster-method",
        choices=["kmeans", "agglomerative", "spectral", "kmeans_constrained", "gmm", "birch"],
        default="kmeans",
        help="Water clustering method (default: kmeans)",
    )
    bio_group.add_argument(
        "--add-implicit-hydrogens",
        action="store_true",
        help="Add missing hydrogens (requires pdbfixer/openmm)",
    )
    bio_group.add_argument(
        "--no-infer-bonds", action="store_true", help="Disable inferred interfragment bonds"
    )
    bio_group.add_argument(
        "--random-state", type=int, default=42, help="Random seed for clustering (default: 42)"
    )
    bio_group.add_argument(
        "--ph", type=float, default=7.4, help="pH for charge calculation (default: 7.4)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    parser.set_defaults(func=_cmd_bio)


def _add_info_parser(subparsers) -> None:
    """Add the 'info' subcommand."""
    parser = subparsers.add_parser(
        "info",
        help="Show version and feature information",
        description="Display version, available features, and optional dependencies.",
    )
    parser.set_defaults(func=_cmd_info)


def _add_geometric_parser(subparsers) -> None:
    """Add the 'geometric' subcommand."""
    from autofragment.cli.partition import add_tiling_options

    parser = subparsers.add_parser(
        "geometric",
        help="Partition using geometric tiling",
        description="Partition a system into tiling-based fragments.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input XYZ file")
    parser.add_argument("--output", "-o", help="Output JSON file (default: input.json)")
    add_tiling_options(parser)
    parser.add_argument(
        "--xyz-units",
        choices=["angstrom", "bohr"],
        default="angstrom",
        help="Units of XYZ coordinates (default: angstrom)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.set_defaults(func=_cmd_geometric)


def _resolve_init_strategy_arg(strategy_name, init_axis):
    """Convert CLI seeding args to an init value for partition_labels.

    Returns None (default k-means++), a strategy name string, or a dict
    with extra kwargs when the ``axis`` strategy needs a specific axis.
    """
    if strategy_name is None:
        return None
    if strategy_name == "axis" and init_axis is not None:
        return {"strategy": "axis", "axis": init_axis}
    return strategy_name


def _cmd_single(args) -> int:
    """Execute the 'single' command."""
    from autofragment import io
    from autofragment.partitioners.molecular import MolecularPartitioner, PartitionError

    # Set default output
    if not args.output:
        args.output = Path(args.input).with_suffix(".json")

    # Read input
    try:
        xyz_kwargs = _xyz_read_kwargs(args)
        system = io.read_xyz(args.input, xyz_units=args.xyz_units, **xyz_kwargs)
    except io.ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    molecules = system.to_molecules(require_metadata=True)
    print(f"Parsed {len(molecules)} molecules from {args.input}")

    # Resolve seeding strategies
    init_axis = getattr(args, "init_axis", None)
    init_strategy = _resolve_init_strategy_arg(
        getattr(args, "init_strategy", None), init_axis
    )
    init_primary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_primary", None), init_axis
    )
    init_secondary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_secondary", None), init_axis
    )
    init_tertiary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_tertiary", None), init_axis
    )

    tiers = getattr(args, "tiers", None)

    # Create partitioner and partition
    try:
        if tiers is not None:
            # Tiered mode
            n_primary = getattr(args, "n_primary", None)
            n_secondary = getattr(args, "n_secondary", None)
            n_tertiary = getattr(args, "n_tertiary", None)

            # Auto-infer n_tertiary for 3-tier mode
            if tiers == 3 and n_tertiary is None:
                n_mol = len(molecules)
                if n_primary and n_secondary:
                    denom = n_primary * n_secondary
                    if n_mol % denom != 0:
                        print(
                            f"Error: Cannot infer n_tertiary: {n_mol} molecules / "
                            f"({n_primary} * {n_secondary}) is not even",
                            file=sys.stderr,
                        )
                        return 1
                    n_tertiary = n_mol // denom
                    print(f"Inferred n_tertiary = {n_tertiary}")

            partitioner = MolecularPartitioner(
                tiers=tiers,
                n_primary=n_primary,
                n_secondary=n_secondary,
                n_tertiary=n_tertiary,
                method=args.method,
                random_state=args.random_state,
                init_strategy=init_strategy,
                init_strategy_primary=init_primary,
                init_strategy_secondary=init_secondary,
                init_strategy_tertiary=init_tertiary,
            )
        else:
            # Flat mode
            partitioner = MolecularPartitioner(
                n_fragments=args.n_fragments,
                method=args.method,
                random_state=args.random_state,
                init_strategy=init_strategy,
            )
        tree = partitioner.partition(system, source_file=args.input)
    except (ValueError, PartitionError) as e:
        print(f"Error during partitioning: {e}", file=sys.stderr)
        return 1

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_json(output_path)
    print(f"Wrote: {output_path}")

    _print_summary(tree)
    return 0


def _cmd_batch(args) -> int:
    """Execute the 'batch' command."""
    from autofragment import io
    from autofragment.partitioners.batch import BatchPartitioner, BatchPartitionError
    from autofragment.partitioners.molecular import MolecularPartitioner, PartitionError

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read reference
    try:
        xyz_kwargs = _xyz_read_kwargs(args)
        ref_system = io.read_xyz(args.reference, xyz_units=args.xyz_units, **xyz_kwargs)
    except io.ValidationError as e:
        print(f"Error parsing reference: {e}", file=sys.stderr)
        return 1

    ref_molecules = ref_system.to_molecules(require_metadata=True)
    print(f"Reference: {len(ref_molecules)} molecules from {args.reference}")

    # Resolve seeding strategies
    init_axis = getattr(args, "init_axis", None)
    init_strategy = _resolve_init_strategy_arg(
        getattr(args, "init_strategy", None), init_axis
    )
    init_primary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_primary", None), init_axis
    )
    init_secondary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_secondary", None), init_axis
    )
    init_tertiary = _resolve_init_strategy_arg(
        getattr(args, "init_strategy_tertiary", None), init_axis
    )

    tiers = getattr(args, "tiers", None)

    # Build reference partition
    try:
        if tiers is not None:
            n_primary = getattr(args, "n_primary", None)
            n_secondary = getattr(args, "n_secondary", None)
            n_tertiary = getattr(args, "n_tertiary", None)

            # Auto-infer n_tertiary for 3-tier mode
            if tiers == 3 and n_tertiary is None:
                n_mol = len(ref_molecules)
                if n_primary and n_secondary:
                    denom = n_primary * n_secondary
                    if n_mol % denom != 0:
                        print("Error: Cannot infer n_tertiary", file=sys.stderr)
                        return 1
                    n_tertiary = n_mol // denom
                    print(f"Inferred n_tertiary = {n_tertiary}")

            partitioner = MolecularPartitioner(
                tiers=tiers,
                n_primary=n_primary,
                n_secondary=n_secondary,
                n_tertiary=n_tertiary,
                method=args.method,
                random_state=args.random_state,
                init_strategy=init_strategy,
                init_strategy_primary=init_primary,
                init_strategy_secondary=init_secondary,
                init_strategy_tertiary=init_tertiary,
            )
        else:
            partitioner = MolecularPartitioner(
                n_fragments=args.n_fragments,
                method=args.method,
                random_state=args.random_state,
                init_strategy=init_strategy,
            )
        batch = BatchPartitioner.from_partitioner(
            partitioner,
            ref_system,
            metric=args.metric,
            hybrid_alpha=args.hybrid_alpha,
            align=args.align,
            source_file=args.reference,
        )
    except (ValueError, PartitionError) as e:
        print(f"Error building reference partition: {e}", file=sys.stderr)
        return 1

    print(f"Using '{args.metric}' metric for matching")
    if args.align:
        print("Kabsch alignment enabled")

    # Collect input files
    input_paths = _collect_inputs(args.input_dir, args.inputs, args.pattern)
    if not input_paths:
        print("Error: no input files found. Provide --input-dir or --inputs.", file=sys.stderr)
        return 1
    print(f"Processing {len(input_paths)} files")

    # Process each file
    success_count = 0
    seen_names: dict[str, int] = {}
    for in_path in input_paths:
        try:
            tgt_system = io.read_xyz(in_path, xyz_units=args.xyz_units, **xyz_kwargs)
        except io.ValidationError as e:
            print(f"[ERROR] Skipping {in_path}: {e}", file=sys.stderr)
            continue

        tgt_molecules = tgt_system.to_molecules(require_metadata=True)
        if len(tgt_molecules) != len(ref_molecules):
            print(
                f"[ERROR] Skipping {in_path}: molecule count mismatch "
                f"({len(tgt_molecules)} vs {len(ref_molecules)})",
                file=sys.stderr,
            )
            continue

        try:
            tree = batch.partition(
                tgt_system,
                source_file=str(in_path),
                force=args.force,
            )
        except BatchPartitionError as e:
            print(f"[ERROR] Skipping {in_path}: {e}", file=sys.stderr)
            continue

        # Write output, disambiguating collisions
        output_name = Path(in_path).with_suffix(".json").name
        if output_name in seen_names:
            seen_names[output_name] += 1
            stem = Path(output_name).stem
            output_name = f"{stem}_{seen_names[output_name]}.json"
            print(f"Warning: name collision, writing as {output_name}", file=sys.stderr)
        else:
            seen_names[output_name] = 0
        out_path = out_dir / output_name
        tree.to_json(out_path)
        print(f"Wrote: {out_path}")
        success_count += 1

    print(f"\nProcessed {success_count}/{len(input_paths)} files successfully")
    return 0 if success_count == len(input_paths) else 1


def _cmd_bio(args) -> int:
    """Execute the 'bio' command."""
    try:
        from autofragment.io.mmcif import MmcifParseError
        from autofragment.partitioners.bio import BioPartitioner
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Bio mode requires optional dependencies. Install with:", file=sys.stderr)
        print("  pip install autofragment[bio]", file=sys.stderr)
        return 1

    # Set default output
    if not args.output:
        args.output = Path(args.input).with_suffix(".json")

    # Create partitioner
    partitioner = BioPartitioner(
        water_clusters=args.water_clusters,
        water_cluster_method=args.water_cluster_method,
        random_state=args.random_state,
        infer_bonds=not args.no_infer_bonds,
        ph=args.ph,
    )

    # Partition
    try:
        tree = partitioner.partition_file(
            args.input,
            add_hydrogens=args.add_implicit_hydrogens,
        )
    except MmcifParseError as e:
        print(f"Error parsing mmCIF: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during partitioning: {e}", file=sys.stderr)
        return 1

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_json(output_path)
    print(f"Wrote: {output_path}")

    _print_summary(tree)
    return 0


def _cmd_info(args) -> int:
    """Execute the 'info' command."""
    from autofragment._version import __version__

    print(f"autofragment {__version__}")
    print()

    # Check dependencies
    print("Core dependencies:")
    print("  numpy: ", end="")
    try:
        import numpy

        print(f"{numpy.__version__}")
    except ImportError:
        print("NOT INSTALLED")

    print("  scikit-learn: ", end="")
    try:
        sklearn_version = importlib.metadata.version("scikit-learn")
        print(sklearn_version)
    except importlib.metadata.PackageNotFoundError:
        print("NOT INSTALLED")

    print()
    print("Optional dependencies:")

    print("  k-means-constrained: ", end="")
    if importlib.util.find_spec("k_means_constrained") is not None:
        print("installed (balanced clustering)")
    else:
        print("NOT INSTALLED (install with: pip install k-means-constrained --no-deps)")

    print("  gemmi: ", end="")
    try:
        gemmi_version = importlib.metadata.version("gemmi")
        print(f"{gemmi_version} (mmCIF parsing)")
    except importlib.metadata.PackageNotFoundError:
        print("NOT INSTALLED (install with: pip install gemmi)")

    print("  pdbfixer: ", end="")
    if importlib.util.find_spec("pdbfixer") is not None:
        print("installed (hydrogen addition)")
    else:
        print("NOT INSTALLED (install with: conda install pdbfixer)")

    return 0


def _cmd_geometric(args) -> int:
    """Execute the 'geometric' command."""
    from autofragment import io
    from autofragment.partitioners.geometric import TilingPartitioner

    if not args.output:
        args.output = Path(args.input).with_suffix(".json")

    try:
        xyz_kwargs = _xyz_read_kwargs(args)
        system = io.read_xyz(args.input, xyz_units=args.xyz_units, **xyz_kwargs)
    except io.ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        partitioner = TilingPartitioner(
            tiling_shape=args.tiling_shape,
            n_fragments=args.n_fragments,
            scale=args.tiling_scale,
            bounds_strategy=args.bounds_strategy,
        )
        tree = partitioner.partition(system, source_file=args.input)
    except ValueError as e:
        print(f"Error during partitioning: {e}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_json(output_path)
    print(f"Wrote: {output_path}")

    _print_summary(tree)
    return 0


def _xyz_read_kwargs(args) -> dict:
    """Build optional kwargs for io.read_xyz from parsed CLI args."""
    kwargs: dict = {}
    atoms_per_mol = getattr(args, "atoms_per_molecule", None)
    if atoms_per_mol is not None:
        kwargs["atoms_per_molecule"] = atoms_per_mol
    if getattr(args, "no_validate_water", False):
        kwargs["validate_water"] = False
    return kwargs


def _collect_inputs(
    input_dir: Optional[str],
    inputs: Optional[List[str]],
    pattern: str,
) -> List[str]:
    """Collect input file paths."""
    if inputs:
        return [str(Path(p)) for p in inputs]
    if input_dir is None:
        return []
    base = Path(input_dir)
    paths = sorted(str(p) for p in base.glob(pattern) if p.is_file())
    return paths


def _print_summary(tree) -> None:
    """Print a summary of the fragment tree."""
    if not tree.fragments:
        return

    print("\nPartition summary:")

    if tree._is_hierarchical:
        print(f"Primary fragments: {len(tree.fragments)}")
        for primary in tree.fragments:
            n_secondary = len(primary.fragments)
            print(f"- {primary.id}: secondary={n_secondary}")
            for secondary in primary.fragments:
                n_tertiary = len(secondary.fragments)
                if n_tertiary > 0:
                    print(f"  - {secondary.id}: tertiary={n_tertiary}")
    else:
        print(f"Fragments: {len(tree.fragments)}")

    if tree.interfragment_bonds:
        print(f"Interfragment bonds: {len(tree.interfragment_bonds)}")


if __name__ == "__main__":
    sys.exit(main())
