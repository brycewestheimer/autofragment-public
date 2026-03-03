# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for fragmentation workflows."""
from pathlib import Path

from autofragment import ChemicalSystem, io
from autofragment.partitioners import BioPartitioner, MolecularPartitioner
from autofragment.partitioners.geometric import UnitCellPartitioner


class TestWaterClusterWorkflow:
    """Test water cluster MBE workflow."""

    def test_water_monomer_fragmentation(self):
        """Test fragmenting water cluster into monomers."""
        # water3.xyz has 3 waters
        data_path = Path(__file__).parent.parent / "data/water3.xyz"

        system = io.read_xyz(str(data_path))

        partitioner = MolecularPartitioner(n_fragments=3)
        tree = partitioner.partition(system)

        # Should be 3 water molecules
        assert tree.n_fragments == 3
        # Each fragment should have 3 atoms (H2O)
        assert all(f.n_atoms == 3 for f in tree.fragments)


class TestProteinWorkflow:
    """Test protein fragmentation workflow."""

    def test_pdb_to_fmo(self, tmp_path):
        """Test PDB -> fragmentation -> FMO input."""
        data_path = Path(__file__).parent.parent / "data/protein.cif"

        # BioPartitioner(residues_per_fragment=1) should yield 2 fragments.
        partitioner = BioPartitioner()
        tree = partitioner.partition_file(str(data_path))

        assert tree.n_fragments == 2

        # Write FMO input functions
        out_file = tmp_path / "protein.inp"
        io.write_gamess_fmo(tree.fragments, out_file, method="MP2", basis="6-31G*")

        assert out_file.exists()
        content = out_file.read_text()
        assert "$FMO" in content
        assert "NFRAG=2" in content
        assert "MP2" in content


class TestMOFWorkflow:
    """Test MOF fragmentation workflow."""

    def test_mof_fragmentation_workflow(self):
        """Test periodic system fragmentation."""
        # Manually create a system since we don't have a CIF reader verified for this context
        from autofragment.core.lattice import Lattice
        from autofragment.core.types import Atom

        lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
        # Create atoms in different octants
        atoms = [
            Atom("Zn", [1, 1, 1]), # (0,0,0)
            Atom("O", [6, 6, 6])   # (1,1,1) if grid is 2x2x2
        ]
        system = ChemicalSystem(atoms=atoms, lattice=lat)

        # Partition 2x2x2
        partitioner = UnitCellPartitioner(grid_shape=(2, 2, 2))
        tree = partitioner.partition(system)

        assert tree.n_fragments > 0
        # Check metadata
        assert tree.fragments[0].metadata["type"] == "unit_cell"
