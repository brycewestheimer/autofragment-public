# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Periodic boundary condition utilities."""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from autofragment.core.lattice import Lattice
    from autofragment.core.types import ChemicalSystem

def minimum_image_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice: "Lattice"
) -> float:
    """Calculate minimum image distance between two points.

    Finds the shortest distance considering periodic replicas.

    Args:
        pos1, pos2: Cartesian coordinates
        lattice: Lattice object

    Returns:
        Minimum distance in Angstrom
    """
    # Convert to fractional coordinates
    frac1 = lattice.cartesian_to_fractional(pos1)
    frac2 = lattice.cartesian_to_fractional(pos2)

    # Difference in fractional coordinates
    diff = frac2 - frac1

    # Apply minimum image: wrap to [-0.5, 0.5)
    diff = diff - np.round(diff)

    # Convert back to Cartesian
    cart_diff = lattice.fractional_to_cartesian(diff)

    return float(np.linalg.norm(cart_diff))


def minimum_image_vector(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice: "Lattice"
) -> np.ndarray:
    """Get minimum image displacement vector from pos1 to pos2.

    Args:
        pos1: Origin point (Cartesian)
        pos2: Target point (Cartesian)
        lattice: Lattice object

    Returns:
        Vector pointing from pos1 to nearest image of pos2
    """
    frac1 = lattice.cartesian_to_fractional(pos1)
    frac2 = lattice.cartesian_to_fractional(pos2)
    diff = frac2 - frac1
    diff = diff - np.round(diff)
    return lattice.fractional_to_cartesian(diff)


def all_distances_periodic(
    positions: np.ndarray,
    lattice: "Lattice"
) -> np.ndarray:
    """Calculate all pairwise minimum image distances.

    Args:
        positions: Nx3 array of Cartesian coordinates
        lattice: Lattice object

    Returns:
        NxN matrix of distances
    """
    n = len(positions)
    distances = np.zeros((n, n))

    # Todo: Vectorize this for performance if needed
    for i in range(n):
        for j in range(i + 1, n):
            d = minimum_image_distance(positions[i], positions[j], lattice)
            distances[i, j] = d
            distances[j, i] = d

    return distances


def get_bond_cutoff(
    elem1: str,
    elem2: str,
    cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """Get bond cutoff distance for an element pair."""
    if cutoffs:
        pair1 = (elem1, elem2)
        pair2 = (elem2, elem1)
        if pair1 in cutoffs:
            return cutoffs[pair1]
        if pair2 in cutoffs:
            return cutoffs[pair2]

    # Fallback to covalent radii + tolerance
    tolerance = 0.4
    try:
        from autofragment.core.bonds import get_covalent_radius
        r1 = get_covalent_radius(elem1)
        r2 = get_covalent_radius(elem2)
        return r1 + r2 + tolerance
    except ValueError:
        return 0.0


def detect_periodic_bonds(
    system: "ChemicalSystem",
    distance_cutoffs: Optional[Dict[Tuple[str, str], float]] = None
) -> List[Tuple[int, int, np.ndarray]]:
    """Detect bonds considering periodic boundaries.

    Returns:
        List of (atom1, atom2, image_vector) where image_vector
        is (i,j,k) indicating which periodic image atom2 is in.
    """
    if not system.periodic or system.lattice is None:
        # Convert existing bonds to format (u, v, zero_vector)
        # Note: system.bonds is List[Dict]
        bonds = []
        for bond in system.bonds:
            u, v = bond["atom1"], bond["atom2"]
            bonds.append((u, v, np.zeros(3)))
        return bonds

    bonds = []
    lattice = system.lattice

    for i, atom1 in enumerate(system.atoms):
        for j, atom2 in enumerate(system.atoms):
            if j <= i:
                continue

            # Check all neighboring cells
            # Optimization: could check minimum image distance first
            min_dist = minimum_image_distance(atom1.coords, atom2.coords, lattice)
            cutoff = get_bond_cutoff(atom1.symbol, atom2.symbol, distance_cutoffs)

            if min_dist > cutoff:
                continue

            # If within cutoff, identify which image
            # Vector from atom1 to nearest image of atom2
            vec = minimum_image_vector(atom1.coords, atom2.coords, lattice)

            # Reconstruct image index
            # vec = (pos2 + lattice_vec) - pos1
            # lattice_vec = vec + pos1 - pos2
            # lattice_vec_frac = lattice.cartesian_to_fractional(lattice_vec_cart)
            # Round to nearest integer to get image indices
            # However, minimum_image_vector returns diff in [ -0.5, 0.5 ]
            # Wait, minimum_image_vector implements:
            # diff = frac2 - frac1
            # diff = diff - round(diff)
            # return lattice.fractional_to_cartesian(diff)

            # So actual displacement = vec
            # Actual pos2 image = pos1 + vec
            # Image shift = (pos1 + vec) - pos2
            # fractional shift
            shift_cart = (atom1.coords + vec) - atom2.coords
            shift_frac = lattice.cartesian_to_fractional(shift_cart)
            image = np.round(shift_frac).astype(int)

            if np.linalg.norm(vec) < cutoff:
                bonds.append((i, j, image))

    return bonds


class PeriodicGraph:
    """Graph representation for periodic systems.

    Edges store image vectors to track which cell neighbor.
    """

    def __init__(self, system: "ChemicalSystem"):
        """Initialize a new PeriodicGraph instance."""
        self.system = system
        self.edges = detect_periodic_bonds(system)

    def get_neighbors(self, atom_idx: int) -> List[Tuple[int, np.ndarray]]:
        """Get neighbors of an atom with their image vectors."""
        neighbors = []
        for i, j, image in self.edges:
            if i == atom_idx:
                neighbors.append((j, image))
            elif j == atom_idx:
                # If i is neighbor of j at image I, then j sees i at image -I
                neighbors.append((i, -image))
        return neighbors


def make_supercell(
    system: "ChemicalSystem",
    repetitions: Tuple[int, int, int]
) -> "ChemicalSystem":
    """Create supercell by repeating unit cell.

    Args:
        system: Original periodic system
        repetitions: (na, nb, nc) repetitions along a, b, c

    Returns:
        New ChemicalSystem with supercell
    """
    from autofragment.core.lattice import Lattice
    from autofragment.core.types import Atom, ChemicalSystem

    if not system.periodic or system.lattice is None:
        raise ValueError("System must be periodic to create supercell")

    na, nb, nc = repetitions

    # New lattice vectors
    new_vectors = system.lattice.vectors.copy()
    new_vectors[0] *= na
    new_vectors[1] *= nb
    new_vectors[2] *= nc
    new_lattice = Lattice(new_vectors)

    # Replicate atoms
    new_atoms = []
    atom_mapping = {}  # (orig_idx, i, j, k) -> new_idx

    new_idx = 0

    # Use detected periodic bonds
    periodic_bonds = detect_periodic_bonds(system)

    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                shift = system.lattice.fractional_to_cartesian(
                    np.array([i, j, k])
                )
                for orig_idx, atom in enumerate(system.atoms):
                    new_coords = atom.coords + shift
                    new_atom = Atom(
                        symbol=atom.symbol,
                        coords=new_coords,
                        charge=atom.charge
                    )
                    new_atoms.append(new_atom)
                    atom_mapping[(orig_idx, i, j, k)] = new_idx
                    new_idx += 1

    # Replicate bonds
    new_bonds = []
    bond_hashes = set()

    for a1, a2, image in periodic_bonds:
        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    # Source atom in this replica
                    src_idx = atom_mapping[(a1, i, j, k)]

                    # Target atom image indices
                    ti = i + image[0]
                    tj = j + image[1]
                    tk = k + image[2]

                    # Wrapped indices (supercell periodicity)
                    ti_wrapped = ti % na
                    tj_wrapped = tj % nb
                    tk_wrapped = tk % nc

                    target_idx = atom_mapping[(a2, ti_wrapped, tj_wrapped, tk_wrapped)]

                    # Avoid duplicates
                    bond_pair = tuple(sorted((src_idx, target_idx)))
                    if bond_pair not in bond_hashes:
                        bond_hashes.add(bond_pair)
                        new_bonds.append({"atom1": src_idx, "atom2": target_idx, "order": 1.0})

    supercell = ChemicalSystem(atoms=new_atoms, bonds=new_bonds)
    supercell.lattice = new_lattice

    return supercell
