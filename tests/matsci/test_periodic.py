# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for periodic system handling."""
import numpy as np

from autofragment.core.lattice import Lattice
from autofragment.core.periodic import (
    PeriodicGraph,
    all_distances_periodic,
    detect_periodic_bonds,
    minimum_image_distance,
    minimum_image_vector,
)
from autofragment.core.types import Atom, ChemicalSystem


def test_lattice_creation_cubic():
    """Test creating a cubic lattice."""
    a = 10.0
    lat = Lattice.from_parameters(a, a, a, 90, 90, 90)

    assert np.allclose(lat.vectors[0], [10, 0, 0])
    assert np.allclose(lat.vectors[1], [0, 10, 0])
    assert np.allclose(lat.vectors[2], [0, 0, 10])
    assert np.isclose(lat.volume, 1000.0)

def test_lattice_creation_triclinic():
    """Test creating a triclinic lattice."""
    lat = Lattice.from_parameters(10, 10, 10, 60, 60, 60)

    # Check volume for this specific case (a^3 * sqrt(1 - 3*cos^2 + 2*cos^3))
    # cos(60) = 0.5
    # 1 - 3*(0.25) + 2*(0.125) = 1 - 0.75 + 0.25 = 0.5
    # vol = 1000 * sqrt(0.5) = 1000 * 0.7071
    expected_vol = 1000 * np.sqrt(0.5)
    assert np.isclose(lat.volume, expected_vol)

def test_coordinates_conversion():
    """Test coordinate conversions."""
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)

    # 0.5, 0.5, 0.5 -> 5.0, 5.0, 5.0
    frac = np.array([0.5, 0.5, 0.5])
    cart = lat.fractional_to_cartesian(frac)
    assert np.allclose(cart, [5.0, 5.0, 5.0])

    back = lat.cartesian_to_fractional(cart)
    assert np.allclose(back, frac)

def test_reciprocal_lattice():
    """Test reciprocal lattice calculation."""
    # Cubic lattice a=2*pi
    # Reciprocal should be identity (if a=1 then recip=2pi, if a=2pi then recip=1)
    # definition: b_i . a_j = 2pi delta_ij

    a = 1.0
    lat = Lattice.from_parameters(a, a, a, 90, 90, 90)
    recip = lat.reciprocal

    # For cubic: b = 2pi/a * I
    expected = 2 * np.pi * np.eye(3)
    assert np.allclose(recip.vectors, expected)

def test_minimum_image_cubic():
    """Test minimum image in cubic cell."""
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)

    # Points at 1.0 and 9.0 along x
    # Distance should be 2.0 (across boundary) not 8.0
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([9.0, 0.0, 0.0])

    dist = minimum_image_distance(p1, p2, lat)
    assert np.isclose(dist, 2.0)

    # Vector from p1 to p2 should point to -2.0 (left)
    # p2 image is at -1.0. Vector = -1.0 - 1.0 = -2.0
    vec = minimum_image_vector(p1, p2, lat)
    assert np.allclose(vec, [-2.0, 0.0, 0.0])

def test_minimum_image_triclinic():
    """Test minimum image in non-orthogonal cell."""
    # Simple skew: a=10, b=10, gamma=60
    # a = [10, 0, 0]
    # b = [5, 5sqrt(3), 0]

    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 60)

    # Point at origin
    p1 = np.array([0.0, 0.0, 0.0])
    # Point near a+b corner
    # a+b = [15, 5sqrt(3), 0]
    # Let's take a point that is close to the origin in one periodic image
    # but far in Cartesian

    # Fractional [0.1, 0.1, 0] vs [0.9, 0.9, 0]
    # Diff is [-0.8, -0.8] -> wraps to [0.2, 0.2]

    p1 = lat.fractional_to_cartesian(np.array([0.1, 0.1, 0.0]))
    p2 = lat.fractional_to_cartesian(np.array([0.9, 0.9, 0.0]))

    dist = minimum_image_distance(p1, p2, lat)

    # Wraps to diff [0.2, 0.2]
    diff_cart = lat.fractional_to_cartesian(np.array([0.2, 0.2, 0.0]))
    expected = np.linalg.norm(diff_cart)

    assert np.isclose(dist, expected)

def test_all_distances():
    """Test pairwise distance matrix."""
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    points = np.array([
        [1.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])

    dists = all_distances_periodic(points, lat)

    assert dists.shape == (3, 3)
    assert np.isclose(dists[0, 1], 2.0) # 1.0 to 9.0 (wrap)
    assert np.isclose(dists[0, 2], 4.0) # 1.0 to 5.0
    assert np.isclose(dists[1, 2], 4.0) # 9.0 to 5.0

def test_periodic_bond_detection():
    """Test detecting bond across boundary."""
    # Atoms at 1.0 and 9.0 in 10.0 cell
    # H-H bond is ~0.74
    # Distance is 2.0. Too far for H-H.

    # Try bonded atoms.
    # d(C-C) ~ 1.54. r(C)=0.76. Cutoff = 1.52 + 0.4 = 1.92

    # Let's place them such that min image distance is 1.5
    # Cell 10x10x10
    # Atom1 at 0.5
    # Atom2 at 9.0 (dist 1.5 across boundary)

    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([0.5, 0.0, 0.0]))
    a2 = Atom("C", np.array([9.0, 0.0, 0.0]))

    sys = ChemicalSystem(atoms=[a1, a2], bonds=[], metadata={})
    sys.lattice = lat

    bonds = detect_periodic_bonds(sys)

    assert len(bonds) == 1
    u, v, image = bonds[0]
    assert u == 0
    assert v == 1
    # a2 is at 9.0. Image at -1.0 is neighbor to 0.5 (dist 1.5)
    # image vector for a2: we need a2 + image*L
    # 9.0 + (-1)*10 = -1.0.
    # So image should be [-1, 0, 0] ?

    # Let's check logic:
    # vec = min_image_vector(a1, a2) -> points from a1 to a2's image
    # a1=0.5. a2=9.0.
    # diff = 9.0 - 0.5 = 8.5. Periodic wrap -> -1.5.
    # vec = -1.5.
    # shift = (a1 + vec) - a2 = (0.5 - 1.5) - 9.0 = -1.0 - 9.0 = -10.0
    # shift in frac = -1.0
    # image = -1.

    assert np.array_equal(image, [-1, 0, 0])

def test_periodic_graph():
    """Test PeriodicGraph class."""
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([0.5, 0.0, 0.0]))
    a2 = Atom("C", np.array([9.0, 0.0, 0.0]))
    sys = ChemicalSystem(atoms=[a1, a2])
    sys.lattice = lat

    pg = PeriodicGraph(sys)
    nbrs = pg.get_neighbors(0)

    assert len(nbrs) == 1
    idx, img = nbrs[0]
    assert idx == 1
    assert np.array_equal(img, [-1, 0, 0])

    nbrs2 = pg.get_neighbors(1)
    idx2, img2 = nbrs2[0]
    assert idx2 == 0
    assert np.array_equal(img2, [1, 0, 0]) # -image

def test_supercell_generation():
    """Test supercell creation."""
    from autofragment.core.periodic import make_supercell

    # 2 atoms bonded across boundary
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([0.5, 0.0, 0.0]))
    a2 = Atom("C", np.array([9.0, 0.0, 0.0]))
    sys = ChemicalSystem(atoms=[a1, a2], bonds=[], metadata={})
    sys.lattice = lat

    # Make 2x1x1 supercell
    sc = make_supercell(sys, (2, 1, 1))

    # New lattice should be 20x10x10
    assert np.allclose(sc.lattice.vectors[0], [20, 0, 0])

    # Should have 4 atoms
    assert len(sc.atoms) == 4

    # Expected bonds:
    # a2_0 (idx 1) -- a1_1 (idx 2)
    # a2_1 (idx 3) -- a1_0 (idx 0) (periodic boundary of supercell)

    assert len(sc.bonds) == 2

    sorted_bonds = sorted([tuple(sorted((b["atom1"], b["atom2"]))) for b in sc.bonds])
    # Pairs should be (1, 2) and (0, 3)
    assert (1, 2) in sorted_bonds
    assert (0, 3) in sorted_bonds
