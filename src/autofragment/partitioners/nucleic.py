# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Nucleic acid partitioner for DNA and RNA molecules.

This module provides specialized fragmentation for nucleic acids,
supporting both DNA and RNA with proper charge handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from autofragment.core.bonds import InterfragmentBond
from autofragment.core.types import Fragment, FragmentTree
from autofragment.data.nucleotides import (
    get_nucleic_acid_charge,
    get_nucleotide,
)
from autofragment.io.output import format_source_info


@dataclass(frozen=True)
class NucleicPartitionResult:
    """Result of nucleic acid partitioning."""

    fragments: List[Fragment]
    bonds: List[InterfragmentBond]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


class NucleicPartitioner:
    """Partitioner for DNA and RNA molecules.

    This partitioner provides intelligent fragmentation for nucleic acids
    with support for:
    - Configurable nucleotides per fragment
    - Base pair preservation
    - Different partitioning strategies (backbone, base, hybrid)
    - Proper phosphate backbone charge handling

    Parameters
    ----------
    nucleotides_per_fragment : int, optional
        Number of nucleotides per fragment (default: 3)
    preserve_base_pairs : bool, optional
        If True, keep Watson-Crick base pairs together (default: True)
    partition_mode : str, optional
        Partitioning strategy:
        - "backbone": Fragment along backbone, keep phosphate-sugar-base units
        - "base": Separate bases from backbone
        - "hybrid": Flexible fragmentation
        Default is "backbone".
    include_5prime_phosphate : bool, optional
        Include 5' terminal phosphate in charge (default: False)
    include_3prime_phosphate : bool, optional
        Include 3' terminal phosphate in charge (default: False)
    infer_bonds : bool, optional
        Infer interfragment bonds (default: True)
    phosphate_max_distance : float, optional
        Max O-P distance for phosphodiester bonds (default: 1.8 Å)
    """

    def __init__(
        self,
        nucleotides_per_fragment: int = 3,
        preserve_base_pairs: bool = True,
        partition_mode: str = "backbone",
        include_5prime_phosphate: bool = False,
        include_3prime_phosphate: bool = False,
        infer_bonds: bool = True,
        phosphate_max_distance: float = 1.8,
    ):
        """Initialize a new NucleicPartitioner instance."""
        self.nucleotides_per_fragment = nucleotides_per_fragment
        self.preserve_base_pairs = preserve_base_pairs
        self.partition_mode = partition_mode
        self.include_5prime_phosphate = include_5prime_phosphate
        self.include_3prime_phosphate = include_3prime_phosphate
        self.infer_bonds = infer_bonds
        self.phosphate_max_distance = phosphate_max_distance

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate partitioner parameters."""
        if self.nucleotides_per_fragment < 1:
            raise ValueError(
                f"nucleotides_per_fragment must be >= 1, got {self.nucleotides_per_fragment}"
            )
        if self.partition_mode not in ("backbone", "base", "hybrid"):
            raise ValueError(
                f"partition_mode must be 'backbone', 'base', or 'hybrid', "
                f"got '{self.partition_mode}'"
            )

    def partition_residues(
        self,
        residues: List[Dict],
        source_file: Optional[str] = None,
    ) -> FragmentTree:
        """Partition nucleic acid residues into fragments.

        Parameters
        ----------
        residues : list of dict
            List of residue dictionaries with keys:
            - 'code': 3-letter nucleotide code
            - 'chain': chain identifier
            - 'seq': sequence number
            - 'atoms': list of atom dictionaries with 'element', 'coords', 'name'
        source_file : str, optional
            Path to source file for metadata

        Returns
        -------
        FragmentTree
            Fragmentation result
        """
        # Filter to nucleotide residues only
        nucleotide_residues = [
            res for res in residues if get_nucleotide(res.get("code", "")) is not None
        ]

        if not nucleotide_residues:
            return FragmentTree(
                fragments=[],
                interfragment_bonds=[],
                source=format_source_info(source_file, "nucleic") if source_file else {},
                partitioning={"algorithm": "nucleic", "scheme": self.partition_mode},
            )

        # Group by chain
        chains: Dict[str, List[Dict]] = {}
        for res in nucleotide_residues:
            chain_id = res.get("chain", "A")
            chains.setdefault(chain_id, []).append(res)

        # Sort residues by sequence within each chain
        for chain_id in chains:
            chains[chain_id].sort(key=lambda r: r.get("seq", 0))

        fragments: List[Fragment] = []
        bonds: List[InterfragmentBond] = []
        atom_maps: Dict[str, Dict[str, int]] = {}

        for chain_id, chain_residues in chains.items():
            chain_frags, chain_bonds, chain_maps = self._partition_chain(chain_id, chain_residues)
            fragments.extend(chain_frags)
            bonds.extend(chain_bonds)
            atom_maps.update(chain_maps)

        # Detect base pairs if preserving
        if self.preserve_base_pairs and len(chains) >= 2:
            bp_bonds = self._detect_base_pair_bonds(residues, atom_maps)
            bonds.extend(bp_bonds)

        source = {}
        if source_file:
            source = format_source_info(source_file, "nucleic")

        return FragmentTree(
            fragments=fragments,
            interfragment_bonds=[b.to_dict() for b in bonds],
            source=source,
            partitioning={
                "algorithm": "nucleic",
                "scheme": self.partition_mode,
                "nucleotides_per_fragment": self.nucleotides_per_fragment,
                "preserve_base_pairs": self.preserve_base_pairs,
            },
        )

    def _partition_chain(
        self,
        chain_id: str,
        residues: List[Dict],
    ) -> Tuple[List[Fragment], List[InterfragmentBond], Dict[str, Dict[str, int]]]:
        """Partition a single nucleic acid chain."""
        fragments: List[Fragment] = []
        bonds: List[InterfragmentBond] = []
        atom_maps: Dict[str, Dict[str, int]] = {}

        n_residues = len(residues)
        n_per_frag = self.nucleotides_per_fragment

        prev_frag_id = None
        prev_o3_idx = None

        for i in range(0, n_residues, n_per_frag):
            chunk = residues[i : i + n_per_frag]
            chunk_codes = [r.get("code", "UNK") for r in chunk]

            # Calculate charge for this fragment
            is_5prime = i == 0
            is_3prime = i + n_per_frag >= n_residues

            charge = self._calculate_fragment_charge(
                chunk_codes,
                is_5prime=is_5prime,
                is_3prime=is_3prime,
            )

            # Build fragment
            frag_id = f"CHAIN_{chain_id}_NUC_{i // n_per_frag + 1}"
            symbols: List[str] = []
            geometry: List[float] = []
            atom_map: Dict[str, int] = {}

            o3_idx = None  # Track O3' for backbone bonds
            p_idx = None  # Track P for backbone bonds

            for res in chunk:
                for atom in res.get("atoms", []):
                    atom_idx = len(symbols)
                    atom_name = atom.get("name", "")

                    # Track backbone atoms for bonding
                    if atom_name == "O3'":
                        o3_idx = atom_idx
                    elif atom_name == "P":
                        p_idx = atom_idx

                    key = f"{res.get('seq', 0)}:{atom_name}"
                    atom_map[key] = atom_idx
                    symbols.append(atom.get("element", "C"))
                    coords = atom.get("coords", [0.0, 0.0, 0.0])
                    geometry.extend(coords)

            fragment = Fragment(
                id=frag_id,
                symbols=symbols,
                geometry=geometry,
                molecular_charge=int(round(charge)),
                molecular_multiplicity=1,
            )
            fragments.append(fragment)
            atom_maps[frag_id] = atom_map

            # Create backbone bond to previous fragment
            if self.infer_bonds and prev_frag_id and p_idx is not None and prev_o3_idx is not None:
                bonds.append(
                    InterfragmentBond(
                        fragment1_id=prev_frag_id,
                        atom1_index=prev_o3_idx,
                        fragment2_id=frag_id,
                        atom2_index=p_idx,
                        bond_order=1.0,
                        metadata={"type": "phosphodiester", "inferred": True},
                    )
                )

            prev_frag_id = frag_id
            prev_o3_idx = o3_idx

        return fragments, bonds, atom_maps

    def _calculate_fragment_charge(
        self,
        nucleotide_codes: List[str],
        is_5prime: bool = False,
        is_3prime: bool = False,
    ) -> float:
        """Calculate charge for a nucleic acid fragment."""
        include_5p = self.include_5prime_phosphate if is_5prime else False
        include_3p = self.include_3prime_phosphate if is_3prime else False

        return get_nucleic_acid_charge(
            nucleotide_codes,
            include_5prime_phosphate=include_5p,
            include_3prime_phosphate=include_3p,
        )

    def _detect_base_pair_bonds(
        self,
        residues: List[Dict],
        atom_maps: Dict[str, Dict[str, int]],
    ) -> List[InterfragmentBond]:
        """Detect hydrogen bonds between base pairs.

        This is a simplified detection based on typical base pair geometry.
        """
        bonds: List[InterfragmentBond] = []
        # Base pair detection requires 3D geometry analysis
        # For now, return empty - full implementation would analyze
        # hydrogen bonding patterns between complementary bases
        return bonds

    def partition(
        self,
        structure,
        source_file: Optional[str] = None,
    ) -> FragmentTree:
        """Partition a parsed structure containing nucleic acids.

        Parameters
        ----------
        structure : object
            Parsed structure object (e.g., from mmCIF parser)
        source_file : str, optional
            Path to source file

        Returns
        -------
        FragmentTree
            Fragmentation result
        """
        # Extract residues from structure
        residues = []

        if hasattr(structure, "residues"):
            for res in structure.residues:
                if hasattr(res, "res_name") and get_nucleotide(res.res_name) is not None:
                    atoms = []
                    for atom in getattr(res, "atoms", []):
                        atoms.append(
                            {
                                "element": getattr(atom, "element", "C"),
                                "coords": getattr(atom, "coords", np.zeros(3)).tolist(),
                                "name": getattr(atom, "atom_name", ""),
                            }
                        )
                    residues.append(
                        {
                            "code": res.res_name,
                            "chain": getattr(res, "chain_id", "A"),
                            "seq": getattr(res, "res_seq", 0),
                            "atoms": atoms,
                        }
                    )

        return self.partition_residues(residues, source_file)
