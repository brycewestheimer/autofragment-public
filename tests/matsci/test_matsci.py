# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for materials science partitioner."""
from typing import List

import numpy as np

from autofragment.core.lattice import Lattice
from autofragment.core.types import Atom, ChemicalSystem, Fragment
from autofragment.partitioners.matsci import MatSciPartitioner


class SimplePartitioner(MatSciPartitioner):
    """Concrete partitioner for testing."""
    def _partition_impl(self, system: ChemicalSystem) -> List[Fragment]:
        # Just put all atoms in one fragment
        symbols = [a.symbol for a in system.atoms]
        coords = []
        for a in system.atoms:
            coords.extend(a.coords.tolist())

        frag = Fragment(
            id="all",
            symbols=symbols,
            geometry=coords
        )
        return [frag]

def test_matsci_partitioner_init():
    p = SimplePartitioner()
    assert not p.use_supercell

def test_partition_chemical_system():
    # Setup system
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([0.0, 0.0, 0.0]))
    sys = ChemicalSystem(atoms=[a1], lattice=lat)

    p = SimplePartitioner()
    tree = p.partition(sys)

    assert tree.n_fragments == 1
    assert tree.fragments[0].n_atoms == 1
    assert tree.fragments[0].metadata["periodic_origin"]

def test_supercell_expansion_in_partition():
    # Setup system
    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([0.0, 0.0, 0.0]))
    sys = ChemicalSystem(atoms=[a1], lattice=lat)

    # 2x1x1 supercell
    p = SimplePartitioner(use_supercell=True, supercell_size=(2, 1, 1))
    tree = p.partition(sys)

    # SimplePartitioner puts all atoms in one fragment
    # Supercell should have 2 atoms (repeated along x)
    frag = tree.fragments[0]
    assert frag.n_atoms == 2
    assert len(frag.symbols) == 2

def test_radial_partitioner():
    from autofragment.partitioners.geometric import RadialPartitioner

    a1 = Atom("C", np.array([0.0, 0.0, 0.0])) # At center
    a2 = Atom("O", np.array([8.0, 0.0, 0.0])) # At radius 8.0
    a3 = Atom("H", np.array([12.0, 0.0, 0.0])) # At radius 12.0

    sys = ChemicalSystem(atoms=[a1, a2, a3])

    # Radii: [5.0, 10.0]
    # Shell 0: < 5.0 -> a1
    # Shell 1: 5.0 <= d < 10.0 -> a2
    # Shell 2: >= 10.0 -> a3

    p = RadialPartitioner(radii=[5.0, 10.0])
    # Manually set center to 0,0,0 to match atoms
    p.center = np.zeros(3)

    tree = p.partition(sys)

    assert tree.n_fragments == 3
    # Check contents
    # Shell order in result corresponds to shells list order if all populated
    # We should check IDs or content

    f0 = next(f for f in tree.fragments if f.metadata["index"] == 0)
    assert f0.n_atoms == 1
    assert f0.symbols[0] == "C"

    f1 = next(f for f in tree.fragments if f.metadata["index"] == 1)
    assert f1.n_atoms == 1
    assert f1.symbols[0] == "O"

    f2 = next(f for f in tree.fragments if f.metadata["index"] == 2)
    assert f2.n_atoms == 1
    assert f2.symbols[0] == "H"

def test_slab_partitioner():
    from autofragment.partitioners.geometric import SlabPartitioner

    lat = Lattice.from_parameters(12, 12, 12, 90, 90, 90)
    # 3 layers along Z (c-axis, index 2). Height 12.
    # Layers: [0, 4), [4, 8), [8, 12)

    a1 = Atom("C", np.array([0.0, 0.0, 2.0]))  # Layer 0 (2.0 < 4.0)
    a2 = Atom("C", np.array([0.0, 0.0, 6.0]))  # Layer 1 (4.0 < 6.0 < 8.0)
    a3 = Atom("C", np.array([0.0, 0.0, 10.0])) # Layer 2 (8.0 < 10.0 < 12.0)

    sys = ChemicalSystem(atoms=[a1, a2, a3], lattice=lat)

    p = SlabPartitioner(axis=2, n_layers=3)
    tree = p.partition(sys)

    assert tree.n_fragments == 3

    assert tree.fragments[0].layer == "0"
    assert tree.fragments[0].n_atoms == 1
    assert tree.fragments[0].symbols == ["C"]

def test_unit_cell_partitioner():
    from autofragment.partitioners.geometric import UnitCellPartitioner

    # Create a 2x1x1 supercell manually
    # Or rely on use_supercell

    lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
    a1 = Atom("C", np.array([5.0, 5.0, 5.0])) # In first cell

    sys = ChemicalSystem(atoms=[a1], lattice=lat)

    # We ask to create a 2x1x1 supercell, then partition it back to 2x1x1 grid
    p = UnitCellPartitioner(
        use_supercell=True,
        supercell_size=(2, 1, 1),
        grid_shape=(2, 1, 1)
    )

    tree = p.partition(sys)

    # make_supercell(2,1,1) -> atoms at 5.0 and 15.0. New lattice 20x10x10.
    # grid_shape (2,1,1) -> divisions at frac 0.5. (Cart 10.0)

    # Atom 1: pos 5.0. Frac in big cell = 5/20 = 0.25.
    # Atom 2: pos 15.0. Frac in big cell = 15/20 = 0.75.

    # 0.25 * 2 = 0.5 -> idx 0
    # 0.75 * 2 = 1.5 -> idx 1

    assert tree.n_fragments == 2

    # Get fragments by grid index
    f0 = next(f for f in tree.fragments if f.metadata["grid_index"] == (0, 0, 0))
    f1 = next(f for f in tree.fragments if f.metadata["grid_index"] == (1, 0, 0))

    assert f0.n_atoms == 1
    assert f1.n_atoms == 1

    # Check original atom is preserved (only coordinates shifted?)
    # Atom 1 is copy of a1.
    pass

def test_surface_partitioner():
    from autofragment.partitioners.geometric import SurfacePartitioner

    # Slab along Z. Vacuum from 0-3 and 17-20 in 20.0 cell?
    # No, box is [0, 20]. max Z atom at 15. min Z atom at 5.
    # Vacuum threshold 3.0.

    # Surface region: [min, min+3) and (max-3, max]
    # min=5. surface < 8.
    # max=15. surface > 12.

    atoms = [
        Atom("H", [0, 0, 5.0]), # min -> Bottom Surface
        Atom("H", [0, 0, 6.0]), # < 8 -> Bottom Surface
        Atom("H", [0, 0, 10.0]), # Bulk
        Atom("H", [0, 0, 14.0]), # > 12 -> Top Surface
        Atom("H", [0, 0, 15.0])  # max -> Top Surface
    ]
    sys = ChemicalSystem(atoms=atoms)

    # We don't need lattice for explicit surface detection if not using fractional,
    # but base MatSciPartitioner might want it for unrelated things.
    # SurfacePartitioner logic uses Cartesian.

    p = SurfacePartitioner(surface_axis=2, surface_depth=3.0)
    tree = p.partition(sys)

    # Should have 3 fragments: top, bottom, bulk
    assert tree.n_fragments == 3

    f_top = next(f for f in tree.fragments if f.metadata["region"] == "surface_top")
    assert f_top.n_atoms == 2

    f_bot = next(f for f in tree.fragments if f.metadata["region"] == "surface_bottom")
    assert f_bot.n_atoms == 2

    f_bulk = next(f for f in tree.fragments if f.metadata["region"] == "bulk")
    assert f_bulk.n_atoms == 1
    assert f_bulk.geometry[2] == 10.0 # Z coordinate
