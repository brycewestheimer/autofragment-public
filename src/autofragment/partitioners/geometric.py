# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric partitioning for materials."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from autofragment.core.lattice import Lattice
from autofragment.core.periodic import minimum_image_distance
from autofragment.core.types import ChemicalSystem, Fragment
from autofragment.geometry.tilings import get_tiling_shape, grid_shape_from_count
from autofragment.io.output import format_partitioning_info, format_source_info
from autofragment.partitioners.matsci import MatSciPartitioner

if TYPE_CHECKING:
    from autofragment.core.types import Atom, FragmentTree, Molecule


def _atoms_center_of_mass(atoms: Sequence["Atom"]) -> np.ndarray:
    """Calculate center of mass from atoms."""
    # Assuming uniform mass for now or based on symbol if we had mass
    # For geometric purposes, geometric center (centroid) is often what is meant unless specified
    # Using centroid:
    if not atoms:
        return np.zeros(3)
    coords = np.array([a.coords for a in atoms])
    return np.mean(coords, axis=0)

class RadialPartitioner(MatSciPartitioner):
    """Partition by radial distance from center."""

    def __init__(
        self,
        center: Optional[np.ndarray] = None,
        radii: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Args:
            center: Center point (defaults to system centroid)
            radii: Shell radii in Angstrom [r1, r2, r3, ...]
                   Defines shells: [0, r1), [r1, r2), ... [rn, inf)
        """
        super().__init__(**kwargs)
        self.center = center
        self.radii = radii or [5.0, 10.0, 15.0]

    def _partition_impl(self, system: "ChemicalSystem") -> List["Fragment"]:
        """Internal helper to partition impl."""
        center = self.center
        if center is None:
            center = _atoms_center_of_mass(system.atoms)

        # Organize shells: list of lists of atom indices
        # Number of shells = len(radii) + 1 (for outer shell)
        shells: List[List[int]] = [[] for _ in range(len(self.radii) + 1)]

        system_lattice = system.lattice if system.periodic else None

        for i, atom in enumerate(system.atoms):
            if system_lattice:
                dist = minimum_image_distance(atom.coords, center, system_lattice)
            else:
                dist = float(np.linalg.norm(atom.coords - center))

            shell_idx = len(self.radii)  # Default to outermost
            for j, r in enumerate(self.radii):
                if dist < r:
                    shell_idx = j
                    break
            shells[shell_idx].append(i)

        fragments = []
        for i, indices in enumerate(shells):
            if not indices:
                continue

            # Extract atoms
            frag_atoms = [system.atoms[idx] for idx in indices]
            # Construct fragment geometry
            coords = []
            symbols = []
            for atom in frag_atoms:
                coords.extend(atom.coords.tolist())
                symbols.append(atom.symbol)

            fragments.append(Fragment(
                id=f"shell_{i}",
                symbols=symbols,
                geometry=coords,
                metadata={"type": "radial_shell", "index": i}
            ))

        return fragments


class SlabPartitioner(MatSciPartitioner):
    """Partition by layers along an axis."""

    def __init__(
        self,
        axis: int = 2,  # 0=a, 1=b, 2=c
        n_layers: int = 3,
        **kwargs
    ):
        """Initialize a new SlabPartitioner instance."""
        super().__init__(**kwargs)
        self.axis = axis
        self.n_layers = n_layers

    def _partition_impl(self, system: "ChemicalSystem") -> List["Fragment"]:
        """Internal helper to partition impl."""
        if not system.periodic or not system.lattice:
            # Need lattice to define layers cleanly along axes even if not periodic,
            # but usually this implies periodic box intuition.
            # If no lattice, cannot do fractional.
            # Fallback: simple Cartesian binning?
            # For now, require lattice.
             raise ValueError("SlabPartitioner requires a system with a lattice.")

        # Convert to fractional, bin by axis
        layers: List[List[int]] = [[] for _ in range(self.n_layers)]

        for i, atom in enumerate(system.atoms):
            frac = system.lattice.cartesian_to_fractional(atom.coords)
            # frac should be roughly in [0, 1) for main cell
            # Map into [0, n_layers)
            val = frac[self.axis]

            # Normalize to [0, 1) to handle PBC wrap if atoms are slightly outside
            val = val - np.floor(val)

            layer_idx = int(val * self.n_layers) % self.n_layers
            layers[layer_idx].append(i)

        fragments = []
        for i, indices in enumerate(layers):
            if not indices:
                continue

            frag_atoms = [system.atoms[idx] for idx in indices]
            coords = []
            symbols = []
            for atom in frag_atoms:
                coords.extend(atom.coords.tolist())
                symbols.append(atom.symbol)

            fragments.append(Fragment(
                id=f"layer_{i}",
                symbols=symbols,
                geometry=coords,
                layer=str(i),
                metadata={"type": "slab_layer", "axis": self.axis}
            ))

        return fragments


class UnitCellPartitioner(MatSciPartitioner):
    """Partition by unit cell boundaries / grid."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (1, 1, 1),
        **kwargs
    ):
        """
        Args:
            grid_shape: Number of divisions along (a, b, c).
                        e.g. (3, 3, 3) divides the cell into 27 blocks.
        """
        super().__init__(**kwargs)
        self.grid_shape = grid_shape

    def _partition_impl(self, system: "ChemicalSystem") -> List["Fragment"]:
        """Internal helper to partition impl."""
        if not system.periodic or not system.lattice:
            raise ValueError("UnitCellPartitioner requires a system with a lattice.")

        fragments_dict: Dict[Tuple[int, int, int], List[int]] = {}  # (ix, iy, iz) -> list of indices

        na, nb, nc = self.grid_shape

        for i, atom in enumerate(system.atoms):
            frac = system.lattice.cartesian_to_fractional(atom.coords)

            # Wrap to [0, 1) to handle numerical noise or shifts
            frac = frac - np.floor(frac)

            # Determine cell index
            ix = int(frac[0] * na) % na
            iy = int(frac[1] * nb) % nb
            iz = int(frac[2] * nc) % nc

            key = (ix, iy, iz)
            if key not in fragments_dict:
                fragments_dict[key] = []
            fragments_dict[key].append(i)

        fragments = []
        # Sort keys for deterministic output
        for key in sorted(fragments_dict.keys()):
            indices = fragments_dict[key]
            symbols = [system.atoms[idx].symbol for idx in indices]
            coords = []
            for idx in indices:
                coords.extend(system.atoms[idx].coords.tolist())

            fragments.append(Fragment(
                id=f"cell_{key[0]}_{key[1]}_{key[2]}",
                symbols=symbols,
                geometry=coords,
                metadata={"type": "unit_cell", "grid_index": key}
            ))

        return fragments


def detect_surface_atoms(
    system: "ChemicalSystem",
    surface_axis: int = 2,  # z-axis for typical slabs
    vacuum_threshold: float = 3.0  # Angstrom
) -> Tuple[List[int], List[int]]:
    """Detect surface vs bulk atoms in slab geometry.

    Returns:
        (surface_atoms, bulk_atoms) as lists of indices
    """
    if not system.atoms:
        return [], []

    positions = np.array([a.coords for a in system.atoms])
    coords = positions[:, surface_axis]

    min_coord = coords.min()
    max_coord = coords.max()

    surface_atoms = []
    bulk_atoms = []

    for i, z in enumerate(coords):
        # Atoms within threshold of the boundaries
        if (z - min_coord < vacuum_threshold) or (max_coord - z < vacuum_threshold):
            surface_atoms.append(i)
        else:
            bulk_atoms.append(i)

    return surface_atoms, bulk_atoms


class SurfacePartitioner(MatSciPartitioner):
    """Partition separating surface from bulk.

    Identifies top and bottom surfaces based on coordinate extremes
    along a specified axis.
    """

    def __init__(
        self,
        surface_axis: int = 2,
        surface_depth: float = 3.0,
        **kwargs
    ):
        """
        Args:
            surface_axis: Axis along which the surface exists (0=x, 1=y, 2=z)
            surface_depth: Depth in Angstroms to consider as surface
        """
        super().__init__(**kwargs)
        self.surface_axis = surface_axis
        self.surface_depth = surface_depth

    def _partition_impl(self, system: "ChemicalSystem") -> List["Fragment"]:
        """Internal helper to partition impl."""
        if not system.atoms:
             return []

        surface_axis = self.surface_axis

        surface_indices, bulk_indices = detect_surface_atoms(
            system, surface_axis, self.surface_depth
        )

        fragments = []

        positions = np.array([a.coords for a in system.atoms])
        coords = positions[:, surface_axis]
        max_coord = coords.max()

        top_surface_indices = []
        bottom_surface_indices = []

        # Split surface into Top and Bottom
        for idx in surface_indices:
            z = coords[idx]
            # Use same threshold logic as detection.
            # However, detection uses min_coord too.
            # If slab is very thin (thickness < 2*depth), overlap can happen or semantics get fuzzy.
            # We assign to closest boundary.

            # Simple check: is it closer to max or min?
            # Or just check if within depth of max.

            if max_coord - z < self.surface_depth:
                top_surface_indices.append(idx)
            else:
                bottom_surface_indices.append(idx)

        # Create fragments
        if top_surface_indices:
             symbols = [system.atoms[i].symbol for i in top_surface_indices]
             geom = []
             for i in top_surface_indices:
                 geom.extend(system.atoms[i].coords.tolist())
             fragments.append(Fragment(
                 id="surface_top",
                 symbols=symbols,
                 geometry=geom,
                 metadata={"region": "surface_top", "axis": surface_axis}
             ))

        if bottom_surface_indices:
             symbols = [system.atoms[i].symbol for i in bottom_surface_indices]
             geom = []
             for i in bottom_surface_indices:
                 geom.extend(system.atoms[i].coords.tolist())
             fragments.append(Fragment(
                 id="surface_bottom",
                 symbols=symbols,
                 geometry=geom,
                 metadata={"region": "surface_bottom", "axis": surface_axis}
             ))

        if bulk_indices:
             symbols = [system.atoms[i].symbol for i in bulk_indices]
             geom = []
             for i in bulk_indices:
                 geom.extend(system.atoms[i].coords.tolist())
             fragments.append(Fragment(
                 id="bulk",
                 symbols=symbols,
                 geometry=geom,
                 metadata={"region": "bulk"}
             ))

        return fragments


def _bounding_box_from_atoms(atoms: Sequence["Atom"]) -> np.ndarray:
    """Internal helper to bounding box from atoms."""
    if not atoms:
        return np.zeros((2, 3))
    coords = np.array([atom.coords for atom in atoms])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return np.vstack([mins, maxs])


def _bounding_box_from_lattice(lattice: Lattice) -> np.ndarray:
    """Internal helper to bounding box from lattice."""
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            lattice.vectors[0],
            lattice.vectors[1],
            lattice.vectors[2],
            lattice.vectors[0] + lattice.vectors[1],
            lattice.vectors[0] + lattice.vectors[2],
            lattice.vectors[1] + lattice.vectors[2],
            lattice.vectors[0] + lattice.vectors[1] + lattice.vectors[2],
        ]
    )
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    return np.vstack([mins, maxs])


def _generate_cell_centers(
    bounds: np.ndarray,
    lattice_vectors: np.ndarray,
    grid_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Internal helper to generate cell centers."""
    origin = bounds[0]
    centers = []
    indices = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                center = (
                    origin
                    + (i + 0.5) * lattice_vectors[0]
                    + (j + 0.5) * lattice_vectors[1]
                    + (k + 0.5) * lattice_vectors[2]
                )
                centers.append(center)
                indices.append((i, j, k))
    return np.array(centers), indices


class TilingPartitioner(MatSciPartitioner):
    """Partition using space-filling tiling cells."""

    def __init__(
        self,
        tiling_shape: str = "cube",
        n_fragments: int = 1,
        scale: float = 1.0,
        bounds_strategy: str = "auto",
        **kwargs,
    ):
        """Initialize a new TilingPartitioner instance."""
        super().__init__(**kwargs)
        if n_fragments <= 0:
            raise ValueError("n_fragments must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.tiling_shape = tiling_shape
        self.n_fragments = n_fragments
        self.scale = scale
        self.bounds_strategy = bounds_strategy

    def partition(
        self,
        system_input: Union[Sequence["Molecule"], ChemicalSystem],
        source_file: str | None = None,
    ) -> "FragmentTree":
        """Partition a chemical system and return the resulting fragment tree."""
        from autofragment.core.types import FragmentTree, molecules_to_system

        if isinstance(system_input, ChemicalSystem):
            system = system_input
        else:
            system = molecules_to_system(system_input)

        if not system.periodic:
            self.treat_as_molecular = True

        if self.use_supercell and system.periodic:
            from autofragment.core.periodic import make_supercell

            system = make_supercell(system, self.supercell_size)

        fragments = self._partition_impl(system)

        source = format_source_info(source_file, "xyz") if source_file else {}
        partitioning = format_partitioning_info(
            algorithm="tiling",
            n_fragments=self.n_fragments,
            tiling_shape=self.tiling_shape,
            scale=self.scale,
            bounds_strategy=self.bounds_strategy,
            periodic=system.periodic,
        )
        return FragmentTree(
            fragments=fragments,
            source=source,
            partitioning=partitioning,
        )

    def _partition_impl(self, system: "ChemicalSystem") -> List["Fragment"]:
        """Internal helper to partition impl."""
        shape = get_tiling_shape(self.tiling_shape)
        lattice_vectors = shape.lattice_vectors(self.scale)
        grid_shape = grid_shape_from_count(self.n_fragments)

        if system.periodic and system.lattice and self.bounds_strategy in ("auto", "lattice"):
            bounds = _bounding_box_from_lattice(system.lattice)
        else:
            bounds = _bounding_box_from_atoms(system.atoms)
            padding = np.array(
                [np.linalg.norm(lattice_vectors[0]),
                 np.linalg.norm(lattice_vectors[1]),
                 np.linalg.norm(lattice_vectors[2])]
            ) / 2.0
            bounds = np.vstack([bounds[0] - padding, bounds[1] + padding])

        centers, center_indices = _generate_cell_centers(bounds, lattice_vectors, grid_shape)

        if len(centers) == 0:
            return []

        fragments_by_cell: Dict[Tuple[int, int, int], List[int]] = {}
        for atom_index, atom in enumerate(system.atoms):
            distances = np.linalg.norm(centers - atom.coords, axis=1)
            center_idx = int(np.argmin(distances))
            cell_index = center_indices[center_idx]
            fragments_by_cell.setdefault(cell_index, []).append(atom_index)

        fragments = []
        for cell_index in sorted(fragments_by_cell.keys()):
            indices = fragments_by_cell[cell_index]
            if not indices:
                continue
            symbols = [system.atoms[idx].symbol for idx in indices]
            coords: List[float] = []
            for idx in indices:
                coords.extend(system.atoms[idx].coords.tolist())
            fragment = Fragment(
                id=f"tile_{cell_index[0]}_{cell_index[1]}_{cell_index[2]}",
                symbols=symbols,
                geometry=coords,
                metadata={
                    "tiling_shape": shape.name,
                    "cell_index": cell_index,
                    "scale": self.scale,
                },
            )
            fragments.append(fragment)

        return fragments
