# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Tiered Hierarchical Fragmentation**: 2-tier and 3-tier recursive partitioning where molecules are first clustered into primary fragments, then sub-clustered into secondary (and optionally tertiary) fragments.
    - `MolecularPartitioner` accepts `tiers=2` or `tiers=3` with `n_primary`, `n_secondary`, and `n_tertiary` parameters.
    - `BatchPartitioner` supports tiered mode with consistent hierarchical labeling across multiple files.
    - `Fragment` now supports child fragments via `fragments` field, with `is_leaf` property and recursive `n_atoms`.
    - `FragmentTree` gains `n_primary`, `_is_hierarchical`, and recursive `n_fragments` properties.
    - Hierarchical fragment naming convention: `PF1`, `PF1_SF1`, `PF1_SF1_TF1`.
    - Full JSON serialization/deserialization roundtrip for hierarchical trees.
- **K-Means Seeding Strategies**: Pluggable initialization for kmeans/kmeans_constrained clustering.
    - Four strategies: `halfplane` (angular sectors in PCA plane), `pca` (PC1 binning), `axis` (Cartesian axis binning), `radial` (XY angular sectors).
    - `compute_seeds()` entry point and `SEEDING_STRATEGIES` registry.
    - Per-tier seeding overrides via `init_strategy_primary`, `init_strategy_secondary`, `init_strategy_tertiary`.
    - `partition_labels()` gains `init` parameter accepting strategy names, numpy arrays, or dicts.
- **Geometric Tiered Partitioning**: `partition_by_planes_tiered()` for angular/z/y-slice hierarchical partitioning.
- **CLI Options**: `--tiers`, `--n-primary`, `--n-secondary`, `--n-tertiary`, `--init-strategy`, `--init-strategy-primary/secondary/tertiary`, `--init-axis` for both `single` and `batch` subcommands.
- **Convenience Function**: `partition_xyz()` accepts optional `tiers`, `n_primary`, `n_secondary`, `n_tertiary`, `init_strategy` parameters.

## [1.0.0] - 2026-02-18

### Added
- Topology-based neighborhood selection utilities with graph-hop and Euclidean layered modes.
- `TopologySelection` for QM/MM region definition and reusable `TopologyNeighborSelection` for broader partitioning workflows.

### Changed
- Updated canonical model alignment: `ChemicalSystem` is the system-level input, partitioners return `Fragment`/`FragmentTree`, and `Molecule` is limited to explicit boundary conversions.
- `read_xyz()` now returns a `ChemicalSystem`; use `read_xyz_molecules()` for isolated molecule lists.
- Added conversion helpers and migration notes for the system/molecule boundary updates.
- Updated user documentation and README for the production v1.0.0 release position, including topology-aware workflow guidance.

## [0.10.0] - 2026-02-02

### Added
- **Examples & Tutorials (Phase 10)**:
    - **Jupyter Notebooks**:
        - `examples/notebooks/01_quickstart.ipynb`: 5-minute introduction to fragmentation concepts.
        - `examples/notebooks/02_water_mbe.ipynb`: Many-Body Expansion (MBE) workflow for water clusters.
        - `examples/notebooks/03_protein_efmo.ipynb`: Protein fragmentation for GAMESS EFMO.
        - `examples/notebooks/04_compare_algorithms.ipynb`: Algorithm benchmarking.
        - `examples/notebooks/05_custom_scoring.ipynb`: Guide to custom scoring and partitioners.
        - `examples/notebooks/06_zeolite_fragment.ipynb`: Periodic system handling for materials.
        - `examples/notebooks/07_qmmm_setup.ipynb`: QM/MM layer assignment and link atoms.
    - **Test Data**:
        - `examples/data/`: `1l2y.pdb` (protein), `sodalite.cif` (zeolite), `aspirin.xyz`/`caffeine.xyz` (drugs), `dna_hairpin.pdb` (nucleic acid).
        - Validated data for all major use cases (Bio, Materials, Solvents).

### Improved
- **BioPartitioner**: Exposed directly in the top-level API (`from autofragment import BioPartitioner`).
- **Fragment Metadata**: Added `n_molecules` property to `Fragment` for easier analysis of molecular clusters.
- **CIF Reader**: Enhanced robustness for `gemmi` integration and lattice parsing.

## [0.9.0] - 2026-02-01

### Added
- **Testing & Quality Assurance (Phase 9)**:
    - **Integration Test Suite**:
        - `tests/integration/test_workflows.py`: End-to-end validation for Water, Protein, and MOF workflows.
        - `tests/integration/test_roundtrip.py`: Verified data integrity for XYZ and PDB file read/write operations.
    - **Property-Based Testing**:
        - Implemented `Hypothesis` strategies for generating random molecular graphs and atoms.
        - Invariant testing for atom conservation and charge balance (`tests/property/test_invariants.py`).
    - **Test Coverage**:
        - Achieved comprehensive coverage across all modules (`core`, `rules`, `algorithms`, `io`, `biological`, `matsci`, `multilevel`).
    - **Documentation**:
        - Updated `docs/development.md` with detailed testing guidelines and suite organization.

### Improved
- **XYZ Reader**: Robust handling of empty comment lines and varying whitespace.
- **UnitCellPartitioner**: Improved import structure for geometric partitioners.
- **BioPartitioner**: Clarified default configuration behavior.

### Fixed
- Fixed critical bug in `read_xyz` where files with empty comment lines caused atom count mismatch errors.

## [0.8.0] - 2026-02-01

### Added
- **Comprehensive Documentation (Phase 8)**:
    - **Theory Documentation**:
        - `docs/theory/clustering.md`: K-means, spectral clustering, hierarchical methods with LaTeX equations.
        - `docs/theory/graph_partitioning.md`: Min-cut, Kernighan-Lin, Louvain, METIS algorithms.
        - `docs/theory/scoring.md`: Scoring function components, weight selection, multi-objective optimization.
        - `docs/theory/mbe.md`: Many-body expansion formulas, convergence properties, BSSE corrections.
        - `docs/theory/qmmm.md`: Embedding schemes, link atoms, ONIOM extrapolation theory.
    - **User Guides**:
        - `docs/guides/quickstart.md`: 5-minute introduction with installation and basic usage.
        - `docs/guides/water_clusters.md`: Complete MBE workflow tutorial for water clusters.
        - `docs/guides/proteins.md`: Protein fragmentation with capping and pH-dependent charge assignment.
        - `docs/guides/materials.md`: MOF, zeolite, and surface fragmentation for materials science.
        - `docs/guides/custom_rules.md`: PatternRule, BondRule, and custom FragmentationRule creation.
        - `docs/guides/output_formats.md`: All QC program writers (GAMESS, Psi4, ORCA, Q-Chem, NWChem, etc.).
    - **Documentation Structure**:
        - Reorganized `docs/index.md` with Getting Started, Tutorials, User Guide, Theory, and API Reference sections.
        - 50+ LaTeX equations with proper mathematical notation.
        - 100+ code examples covering all major workflows.

### Improved
- **Phase 3 Algorithm Documentation**: Expanded the Fragmentation Algorithms guide with detailed usage for Scoring and Global Optimization.
- **Rules Engine API**: Added `RuleSet` and `BondRule` to the public API and enhanced docstring examples.

### Changed
- Updated Sphinx autodoc configuration for cleaner API generation.
- Improved documentation navigation and cross-references.

## [0.7.0] - 2026-02-01

### Added
- **QM/MM & Multi-Level Methods (Phase 7)**:
    - **Multi-Level Layer System**:
        - `ComputationalLayer` dataclass for defining computational regions with method, basis, and atoms.
        - `MultiLevelScheme` container for managing multiple layers with validation.
        - `LayerType` enum (HIGH, MEDIUM, LOW, MM) and `EmbeddingType` enum (ELECTROSTATIC, MECHANICAL, POLARIZABLE).
        - `LinkAtom` dataclass for boundary atom handling.
    - **Layer Assignment Algorithms**:
        - `assign_by_distance()`: Distance-based layer assignment with configurable cutoffs.
        - `assign_by_residue()`: Residue name-based assignment for proteins.
        - `assign_by_element()`: Element-based layer assignment.
        - `assign_by_custom()`: Custom selector function support.
        - `expand_selection_to_residues()`: Expand partial selections to complete residues.
        - `validate_layer_assignment()`: Validation for overlap and coverage.
    - **ONIOM Scheme Support**:
        - `ONIOMScheme` class with parsing from strings like "ONIOM(B3LYP/6-31G*:UFF)".
        - Support for 2-layer and 3-layer ONIOM configurations.
        - `to_gaussian_input()` and `to_gamess_input()` for input generation.
        - `create_oniom_scheme()` convenience function.
    - **QM/MM Partitioner**:
        - `QMMMPartitioner`: Divide systems into QM, buffer, and MM regions.
        - `QMMMResult` dataclass with properties and serialization.
        - **Selection Strategies**:
            - `AtomSelection`: Explicit atom indices.
            - `ResidueSelection`: By residue name or number.
            - `DistanceSelection`: Spherical selection from center point.
            - `CombinedSelection`: Union or intersection of multiple strategies.
        - Automatic bond detection and cut bond identification.
        - Buffer region generation with configurable radius.
        - `partition_to_fragment_tree()` for compatibility with existing workflows.
    - **Link Atom Positioning**:
        - Advanced `link_atoms.py` module with g-factor approach.
        - `calculate_g_factor()` for bond length ratios.
        - `position_link_atom_gfactor()` and `position_link_atom_fixed_distance()`.
        - `LinkAtomInfo` dataclass with force redistribution support.
        - `create_link_atoms_for_cut_bonds()` batch creation.
        - `validate_link_atoms()` for position validation.
    - **Point Charge Embedding**:
        - `PointCharge` dataclass for electrostatic embedding.
        - `PointChargeEmbedding` class for charge generation and formatting.
        - GAMESS ($EFRAG) and Gaussian output formats.
        - Charge redistribution near link atoms to avoid overpolarization.
        - `generate_simple_charge_array()` convenience function.
- **Documentation**:
    - `docs/multilevel.md`: Comprehensive guide for multi-level methods and QM/MM.
- **Tests**:
    - Reorganized multi-level tests into `tests/multilevel/`.
    - 117 new tests covering layers, ONIOM, QM/MM partitioning, link atoms, and point charges.

### Changed
- Updated `autofragment.partitioners` module to export QM/MM classes.
- Updated `autofragment.multilevel` module to export all new classes and functions.

## [0.6.0] - 2026-01-31

### Added
- **Materials Science Support (Phase 6)**:
    - **Lattice Infrastructure**: `Lattice` class for unit cell operations, fractional/Cartesian conversion, and reciprocal lattice.
    - **Periodic Utilities**: Minimum image convention distance calculations and supercell generation (`make_supercell`).
    - **MatSciPartitioner**: Base class for periodic and materials-specific partitioning.
    - **Geometric Partitioners**:
        - `RadialPartitioner`: Shell-based fragmentation.
        - `SlabPartitioner`: Axis-aligned layer fragmentation.
        - `UnitCellPartitioner`: Grid-based domain decomposition.
        - `SurfacePartitioner`: Automated detection and separation of top/bottom surfaces and bulk atoms.
    - **Extended Material Rules**:
        - **Zeolites**: `ZeoliteAcidSiteRule` for maintaining active sites.
        - **MOFs**: `MetalCarboxylateRule` for node/linker separation and `MetalNodeRule`.
        - **Polymers**: `PolymerBackboneRule` and monomer detection heuristics.
        - **Perovskites**: `PerovskiteOctahedralRule` with corner-sharing awareness.
        - **Silica**: `SilanolRule` for surface termination.
    - **Materials Data**: Centralized definitions for SBUs and polymer backbones in `src/autofragment/data/materials.py`.
- **Documentation**:
    - `docs/materials.md`: Comprehensive guide for materials science fragmentation.
- **Tests**:
    - Reorganized materials tests into `tests/matsci/`.
    - 15+ new tests covering periodic boundaries, geometric partitioning, and material rules.

### Changed
- `Fragment` type now supports `metadata` for region-specific labels (e.g., `surface_top`, `grid_index`).
- `to_dict`/`from_dict` serialization updated for new metadata fields.

## [0.5.0] - 2026-01-31

### Added
- **Biological System Enhancements**:
    - **Amino Acid Data**: Complete charge data for all 20 standard amino acids + MSE/SEC.
    - **pH Calculator**: Henderson-Hasselbalch equation for pH-dependent charge calculation.
    - **PTM Database**: Support for phosphorylated, acetylated, and methylated residues.
    - **Nucleotide Data**: Comprehensive definitions for DNA and RNA.
    - **NucleicPartitioner**: Specialized partitioner for nucleic acids with:
        - Configurable nucleotides per fragment.
        - Watson-Crick and Wobble base pair preservation.
        - Backbone, Base, and Hybrid partitioning modes.
        - Accurate phosphate backbone (-1) and terminal charges.
    - **BioPartitioner Upgrade**: Integrated new charge engine and pH awareness.
- **Documentation**:
    - `docs/biological.md`: Guide for biological system partitioning.
    - `docs/algorithms.md`: Guide for fragmentation algorithms (Phase 3).
- **Tests**:
    - 32 new tests covering biological data, pH calculations, and nucleic partitioning.

## [0.4.0] - 2026-01-31

### Added
- **Comprehensive I/O Format Support**:
    - **Readers (11 formats)**:
        - `read_pdb()`: PDB files with ATOM/HETATM, CONECT records, multi-MODEL support.
        - `read_mol2()`: Tripos MOL2 with atom types, bond orders, and partial charges.
        - `read_sdf()` / `read_sdf_multi()`: MDL SDF V2000/V3000 formats.
        - `read_qcschema()`: MolSSI QCSchema JSON with Bohr→Angstrom conversion.
        - `read_gamess_input()`: GAMESS $DATA and $CONTRL parsing.
        - `read_psi4_input()`: Psi4 molecule blocks.
        - `read_qchem_input()`: Q-Chem $molecule sections.
        - `read_orca_input()`: ORCA *xyz blocks.
        - `read_nwchem_input()`: NWChem geometry blocks.
        - `read_poscar()` / `read_contcar()`: VASP POSCAR/CONTCAR with lattice vectors.
        - `read_cif()`: CIF/mmCIF with optional gemmi support.
    - **Writers (12 formats)**:
        - `write_gamess_fmo()`: GAMESS FMO with $FMO, $FMOBND, INDAT.
        - `write_gamess_efmo()` / `write_gamess_efp()`: EFMO and EFP methods.
        - `write_psi4_sapt()` / `write_psi4_fragment()`: Psi4 SAPT and fragment calculations.
        - `write_qchem_efp()` / `write_qchem_xsapt()` / `write_qchem_fragmo()`: Q-Chem methods.
        - `write_nwchem_fragment()` / `write_nwchem_bsse()`: NWChem with BSSE support.
        - `write_orca_fragment()` / `write_orca_multijob()`: ORCA with $new_job.
        - `write_qcschema()`: QCSchema JSON with fragment annotations.
        - `write_molpro_sapt()`: Molpro SAPT/DF-LMP2.
        - `write_turbomole_fragment()`: Turbomole coord files.
        - `write_cfour_fragment()`: CFOUR ZMAT format.
        - `write_xyz_fragments()`: XYZ with fragment markers.
        - `write_pdb_fragments()`: PDB with fragments as chains/models.
    - **Format Registry**:
        - `FormatRegistry` for auto-detection and extensible format handling.
        - `register_reader()` / `register_writer()` for custom formats.
        - `supported_formats()` to list available formats.
- **New Documentation**:
    - Comprehensive I/O Formats guide (`docs/io_formats.md`).
- **New Tests**:
    - 24 new tests for readers and writers.
    - Test coverage for PDB, MOL2, SDF, QCSchema, GAMESS, Psi4, ORCA, VASP formats.

### Changed
- Updated `autofragment.io` module to export all new readers and writers.
- Reorganized I/O tests into `tests/io_tests/` subdirectory.

## [0.3.0] - 2026-01-30


### Added
- **Fragmentation Rules Engine**:
    - `RuleEngine` for managing and applying fragmentation constraints.
    - Priority-based rule evaluation and "most restrictive wins" conflict resolution.
    - `FragmentationRule` abstract base class for custom rule creation.
    - Built-in **Common Rules**: Aromatic rings, double/triple bonds, metal coordination, and functional groups.
    - Built-in **Biological Rules**: Configurable peptide bonds, disulfide bridges, proline rings, and alpha-beta carbon preferences.
    - Built-in **Materials Science Rules**: Siloxane bridges, MOF linkers, metal nodes, and perovskite octahedra.
- **Fragmentation Algorithms**:
    - Comprehensive scoring system: Bond Penalty, Size Variance, Interface Score, Chemical Integrity, Computational Cost.
    - Graph Partitioning: Min-Cut (Stoer-Wagner), Balanced Partitioning (Kernighan-Lin), Community Detection (Louvain), METIS integration.
    - Optimization Engine: Greedy Optimizer with local search, Simulated Annealing framework.
    - Hierarchical decomposition tree structure.
    - ML Interface placeholders for future GNN integration.
- **Core Foundation**:
    - `MolecularGraph` class for robust graph-based molecular representation.
    - Bond order inference logic (Single, Double, Triple, Aromatic).
    - Ring and bridge (cut-edge) detection algorithms.
    - Subgraph extraction capabilities.
- **Chemistry Utilities**:
    - Periodic table data (mass, electronegativity, radii).
    - Formal charge estimation.
    - Aromaticity detection heuristics.
- **Data Structures**:
    - `ChemicalSystem` for managing complex environments.
    - Enhanced `Fragment` type with QM/MM layer metadata.
    - `FragmentationScheme` for reproducibility tracking.
- **Documentation**:
    - New "Core Concepts" guide.
    - Comprehensive guide for the Fragmentation Rules Engine.
    - Expanded API reference with rules module documentation.

### Changed
- Reorganized test suite into `core`, `algorithms`, `interface`, and `rules` subdirectories.
- Added 40 new comprehensive tests for the rules engine.
- Updated `MolecularPartitioner` to use new core types internally (gradual migration).

## [0.1.0] - Initial Release

### Added
- Basic `MolecularPartitioner` with k-means support.
- `partition_xyz` convenience function.
- CLI for single and batch processing.
- Support for XYZ and mmCIF input formats.
