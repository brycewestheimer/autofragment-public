# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Biological system partitioner for mmCIF files.

This module partitions biological structures (proteins, waters, ligands)
into a flat list of fragments (typically residue-level).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from autofragment.algorithms.clustering import partition_labels
from autofragment.chemistry.ph import get_sidechain_charge, validate_ph
from autofragment.core.bonds import COVALENT_RADII, InterfragmentBond
from autofragment.core.types import Fragment, FragmentTree
from autofragment.io.mmcif import (
    MmcifStructure,
    ResidueRecord,
    parse_mmcif,
)
from autofragment.io.output import format_source_info

# Standard amino acid charges at physiological pH (~7.4)
RESIDUE_CHARGES_PH7: Dict[str, int] = {
    "ASP": -1,  # Aspartate
    "GLU": -1,  # Glutamate
    "LYS": +1,  # Lysine
    "ARG": +1,  # Arginine
    "HIS": 0,   # Histidine (mostly neutral at pH 7)
    "HID": 0,
    "HIE": 0,
    "HIP": +1,  # Doubly protonated histidine
}


def get_residue_charge(res_name: str) -> int:
    """Get the expected charge for a residue at physiological pH."""
    return RESIDUE_CHARGES_PH7.get(res_name.strip().upper(), 0)


@dataclass(frozen=True)
class BioPartitionResult:
    """Result of biological partitioning."""

    fragments: List[Fragment]
    bonds: List[InterfragmentBond]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return float(np.linalg.norm(a - b))


def _is_covalent(atom1: str, atom2: str, distance: float, tolerance: float) -> bool:
    """Check if two atoms are covalently bonded based on distance."""
    r1 = COVALENT_RADII.get(atom1.capitalize())
    r2 = COVALENT_RADII.get(atom2.capitalize())
    if r1 is None or r2 is None:
        return False
    return distance <= (r1 + r2 + tolerance)


def _residue_fragment_id(res: ResidueRecord) -> str:
    """Generate a unique fragment ID for a residue."""
    ins = res.ins_code or ""
    return f"{res.chain_id}:{res.res_name}:{res.res_seq}{ins}"


def _residue_key(res: ResidueRecord) -> Tuple[str, int, str, str]:
    """Key that uniquely identifies a residue record."""
    return (res.chain_id, int(res.res_seq), res.ins_code or "", res.res_name.strip())


def _prefixed_id(prefix: str, base_id: str) -> str:
    """Create a stable, readable prefixed fragment id."""
    return f"{prefix}|{base_id}"


def _build_residue_fragment(
    fragment_id: str, res: ResidueRecord, ph: float = 7.4
) -> Tuple[Fragment, Dict[str, int]]:
    """Build a Fragment from a residue."""
    symbols: List[str] = []
    geometry: List[float] = []
    atom_index_map: Dict[str, int] = {}

    for atom in res.atoms:
        atom_index_map.setdefault(atom.atom_name, len(symbols))
        symbols.append(atom.element)
        geometry.extend(atom.coords.tolist())

    charge = round(get_sidechain_charge(res.res_name, ph))

    fragment = Fragment(
        id=fragment_id,
        symbols=symbols,
        geometry=geometry,
        molecular_charge=charge,
        molecular_multiplicity=1,
    )
    return fragment, atom_index_map


def _assign_secondary_structure(
    residues: List[ResidueRecord],
    segments: Sequence,
) -> Dict[Tuple[str, int], str]:
    """Assign secondary structure labels to residues."""
    assignment: Dict[Tuple[str, int], str] = {}

    for segment in segments:
        for seq in range(segment.start_seq, segment.end_seq + 1):
            assignment[(segment.chain_id, seq)] = segment.kind

    for res in residues:
        assignment.setdefault((res.chain_id, res.res_seq), "COIL")

    return assignment


def _segment_secondary_fragments(
    residues: List[ResidueRecord],
    assignment: Dict[Tuple[str, int], str],
) -> List[Tuple[str, List[ResidueRecord]]]:
    """Group residues into secondary structure segments."""
    segments: List[Tuple[str, List[ResidueRecord]]] = []
    if not residues:
        return segments

    current_kind = assignment.get((residues[0].chain_id, residues[0].res_seq), "COIL")
    current: List[ResidueRecord] = []
    kind_counts: Dict[str, int] = {}

    for res in residues:
        kind = assignment.get((res.chain_id, res.res_seq), "COIL")
        if kind != current_kind and current:
            kind_counts[current_kind] = kind_counts.get(current_kind, 0) + 1
            label = f"{current_kind}{kind_counts[current_kind]}"
            segments.append((label, current))
            current = []
            current_kind = kind
        current.append(res)

    if current:
        kind_counts[current_kind] = kind_counts.get(current_kind, 0) + 1
        label = f"{current_kind}{kind_counts[current_kind]}"
        segments.append((label, current))

    return segments


def _infer_peptide_bonds(
    chain_residues: List[ResidueRecord],
    residue_id_map: Dict[Tuple[str, int, str, str], str],
    atom_maps: Dict[str, Dict[str, int]],
    peptide_max_distance: float,
) -> List[InterfragmentBond]:
    """Infer peptide bonds between consecutive residues."""
    bonds: List[InterfragmentBond] = []
    chain_residues = sorted(chain_residues, key=lambda r: (r.res_seq, r.ins_code))

    for res_i, res_j in zip(chain_residues[:-1], chain_residues[1:]):
        if not (res_i.is_polymer and res_j.is_polymer):
            continue
        id_i = residue_id_map.get(_residue_key(res_i))
        id_j = residue_id_map.get(_residue_key(res_j))
        if id_i is None or id_j is None:
            continue
        map_i = atom_maps.get(id_i, {})
        map_j = atom_maps.get(id_j, {})
        if "C" not in map_i or "N" not in map_j:
            continue

        atom_i = res_i.atoms[map_i["C"]]
        atom_j = res_j.atoms[map_j["N"]]
        if _distance(atom_i.coords, atom_j.coords) <= peptide_max_distance:
            bonds.append(
                InterfragmentBond(
                    fragment1_id=id_i,
                    atom1_index=map_i["C"],
                    fragment2_id=id_j,
                    atom2_index=map_j["N"],
                    bond_order=1.0,
                    metadata={"type": "peptide", "inferred": True},
                )
            )

    return bonds


def _infer_disulfide_bonds(
    residues: List[ResidueRecord],
    residue_id_map: Dict[Tuple[str, int, str, str], str],
    atom_maps: Dict[str, Dict[str, int]],
    max_distance: float,
) -> List[InterfragmentBond]:
    """Infer disulfide bonds between cysteine residues."""
    cysteines = [res for res in residues if res.res_name.upper() == "CYS"]
    bonds: List[InterfragmentBond] = []

    for i, res_i in enumerate(cysteines):
        id_i = residue_id_map.get(_residue_key(res_i))
        if id_i is None:
            continue
        map_i = atom_maps.get(id_i, {})
        if "SG" not in map_i:
            continue
        atom_i = res_i.atoms[map_i["SG"]]

        for res_j in cysteines[i + 1:]:
            id_j = residue_id_map.get(_residue_key(res_j))
            if id_j is None:
                continue
            map_j = atom_maps.get(id_j, {})
            if "SG" not in map_j:
                continue
            atom_j = res_j.atoms[map_j["SG"]]

            if _distance(atom_i.coords, atom_j.coords) <= max_distance:
                bonds.append(
                    InterfragmentBond(
                        fragment1_id=id_i,
                        atom1_index=map_i["SG"],
                        fragment2_id=id_j,
                        atom2_index=map_j["SG"],
                        bond_order=1.0,
                        metadata={"type": "disulfide", "inferred": True},
                    )
                )

    return bonds


def _infer_ligand_bonds(
    residues: List[ResidueRecord],
    residue_id_map: Dict[Tuple[str, int, str, str], str],
    atom_maps: Dict[str, Dict[str, int]],
    tolerance: float,
) -> List[InterfragmentBond]:
    """Infer covalent bonds between ligands and polymers."""
    ligands = [res for res in residues if res.is_ligand]
    polymers = [res for res in residues if res.is_polymer]
    bonds: List[InterfragmentBond] = []

    for ligand in ligands:
        ligand_id = residue_id_map.get(_residue_key(ligand))
        if ligand_id is None:
            continue
        ligand_map = atom_maps.get(ligand_id, {})

        for polymer in polymers:
            polymer_id = residue_id_map.get(_residue_key(polymer))
            if polymer_id is None:
                continue
            polymer_map = atom_maps.get(polymer_id, {})
            best_pair = None
            best_dist = None

            for lig_atom in ligand.atoms:
                for poly_atom in polymer.atoms:
                    dist = _distance(lig_atom.coords, poly_atom.coords)
                    if _is_covalent(lig_atom.element, poly_atom.element, dist, tolerance):
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_pair = (lig_atom.atom_name, poly_atom.atom_name)

            if best_pair is None:
                continue

            lig_idx = ligand_map.get(best_pair[0])
            poly_idx = polymer_map.get(best_pair[1])
            if lig_idx is None or poly_idx is None:
                continue

            bonds.append(
                InterfragmentBond(
                    fragment1_id=ligand_id,
                    atom1_index=lig_idx,
                    fragment2_id=polymer_id,
                    atom2_index=poly_idx,
                    bond_order=1.0,
                    metadata={"type": "protein_ligand", "inferred": True},
                )
            )

    return bonds


class BioPartitioner:
    """
    Partitioner for biological systems from mmCIF files.

    This partitioner produces a flat list of fragments (residue-level by
    default), and may prefix fragment IDs based on chain/segment information.

    Parameters
    ----------
    water_clusters : int, optional
        Number of water clusters per chain. Auto-detected if not set.
    water_cluster_method : str, optional
        Clustering method for waters. Default is "kmeans_constrained".
    random_state : int, optional
        Random seed for clustering. Default is 42.
    infer_bonds : bool, optional
        If True, infer interfragment bonds. Default is True.
    peptide_max_distance : float, optional
        Max C-N distance for peptide bonds (Angstroms). Default is 1.8.
    disulfide_max_distance : float, optional
        Max SG-SG distance for disulfide bonds. Default is 2.2.
    ligand_bond_tolerance : float, optional
        Tolerance for ligand-protein bonds. Default is 0.45.
    """

    def __init__(
        self,
        water_clusters: Optional[int] = None,
        water_cluster_method: str = "kmeans_constrained",
        random_state: int = 42,
        infer_bonds: bool = True,
        peptide_max_distance: float = 1.8,
        disulfide_max_distance: float = 2.2,
        ligand_bond_tolerance: float = 0.45,
        ph: float = 7.4,
    ):
        """Initialize a new BioPartitioner instance."""
        validate_ph(ph)
        self.water_clusters = water_clusters
        self.water_cluster_method = water_cluster_method
        self.random_state = random_state
        self.infer_bonds = infer_bonds
        self.peptide_max_distance = peptide_max_distance
        self.disulfide_max_distance = disulfide_max_distance
        self.ligand_bond_tolerance = ligand_bond_tolerance
        self.ph = ph

    def partition_file(
        self,
        filepath: str,
        add_hydrogens: bool = False,
    ) -> FragmentTree:
        """
        Partition a biological system from an mmCIF file.

        Parameters
        ----------
        filepath : str
            Path to mmCIF file.
        add_hydrogens : bool, optional
            If True, add missing hydrogens. Default is False.

        Returns
        -------
        FragmentTree
            Fragmentation result containing a flat fragments list.
        """
        structure = parse_mmcif(filepath, add_hydrogens=add_hydrogens)
        return self.partition(structure, source_file=filepath)

    def partition(
        self,
        structure: MmcifStructure,
        source_file: Optional[str] = None,
    ) -> FragmentTree:
        """
        Partition a parsed mmCIF structure.

        Parameters
        ----------
        structure : MmcifStructure
            Parsed structure.
        source_file : str, optional
            Path to source file for metadata.

        Returns
        -------
        FragmentTree
            Fragmentation result containing a flat fragments list.
        """
        result = self._build_partition(structure)

        fragments = result.fragments
        bonds = [b.to_dict() for b in result.bonds]

        source = {}
        if source_file:
            source = format_source_info(source_file, "mmcif")

        partitioning = {
            "algorithm": "biological",
            "scheme": "flat_residues",
        }

        return FragmentTree(
            fragments=fragments,
            interfragment_bonds=bonds,
            source=source,
            partitioning=partitioning,
        )

    def _build_partition(self, structure: MmcifStructure) -> BioPartitionResult:
        """Build the biological partition."""
        residues = structure.residues
        assignment = _assign_secondary_structure(residues, structure.secondary_segments)

        fragments: List[Fragment] = []
        atom_maps: Dict[str, Dict[str, int]] = {}
        residue_id_map: Dict[Tuple[str, int, str, str], str] = {}

        # Group residues by chain
        residues_by_chain: Dict[str, List[ResidueRecord]] = {}
        for res in residues:
            residues_by_chain.setdefault(res.chain_id, []).append(res)

        for chain_id, chain_residues in residues_by_chain.items():
            waters = [res for res in chain_residues if res.is_water]
            polymers = [res for res in chain_residues if res.is_polymer]
            ligands = [res for res in chain_residues if res.is_ligand]

            # Process waters
            if waters:
                n_waters = len(waters)
                n_clusters = self.water_clusters
                if n_clusters is None or n_clusters <= 0:
                    n_clusters = max(1, int(round(n_waters ** 0.5)))
                n_clusters = min(n_clusters, n_waters)

                cents = np.stack(
                    [np.mean([a.coords for a in res.atoms], axis=0) for res in waters],
                    axis=0,
                )
                labels = partition_labels(
                    cents, n_clusters, self.water_cluster_method, self.random_state
                )

                for res, lbl in zip(waters, labels.tolist()):
                    prefix = f"CHAIN_{chain_id}_WCL{int(lbl) + 1}"
                    frag_id = _prefixed_id(prefix, _residue_fragment_id(res))
                    fragment, atom_map = _build_residue_fragment(frag_id, res, ph=self.ph)
                    fragments.append(fragment)
                    atom_maps[fragment.id] = atom_map
                    residue_id_map[_residue_key(res)] = fragment.id

            # Process polymers
            if polymers:
                ordered = sorted(polymers, key=lambda r: (r.res_seq, r.ins_code))
                segments = _segment_secondary_fragments(ordered, assignment)
                for label, segment_residues in segments:
                    for res in segment_residues:
                        prefix = f"CHAIN_{chain_id}_{label}"
                        frag_id = _prefixed_id(prefix, _residue_fragment_id(res))
                        fragment, atom_map = _build_residue_fragment(frag_id, res, ph=self.ph)
                        fragments.append(fragment)
                        atom_maps[fragment.id] = atom_map
                        residue_id_map[_residue_key(res)] = fragment.id

            if ligands:
                for res in ligands:
                    prefix = f"CHAIN_{chain_id}_LIG"
                    frag_id = _prefixed_id(prefix, _residue_fragment_id(res))
                    fragment, atom_map = _build_residue_fragment(frag_id, res, ph=self.ph)
                    fragments.append(fragment)
                    atom_maps[fragment.id] = atom_map
                    residue_id_map[_residue_key(res)] = fragment.id

        # Infer bonds
        bonds: List[InterfragmentBond] = []
        if self.infer_bonds:
            for chain_id, chain_residues in residues_by_chain.items():
                polymer_residues = [res for res in chain_residues if res.is_polymer]
                if polymer_residues:
                    bonds.extend(
                        _infer_peptide_bonds(
                            polymer_residues,
                            residue_id_map,
                            atom_maps,
                            self.peptide_max_distance,
                        )
                    )

            bonds.extend(
                _infer_disulfide_bonds(
                    residues,
                    residue_id_map,
                    atom_maps,
                    self.disulfide_max_distance,
                )
            )
            bonds.extend(
                _infer_ligand_bonds(
                    residues,
                    residue_id_map,
                    atom_maps,
                    self.ligand_bond_tolerance,
                )
            )

        # Deduplicate bonds
        unique: List[InterfragmentBond] = []
        seen = set()
        for bond in bonds:
            key = tuple(
                sorted(
                    [(bond.fragment1_id, bond.atom1_index), (bond.fragment2_id, bond.atom2_index)]
                )
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(bond)

        return BioPartitionResult(fragments=fragments, bonds=unique)
