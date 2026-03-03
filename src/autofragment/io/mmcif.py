# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal mmCIF parser for biological partitioning.

This module provides parsing for mmCIF files containing biological
structures (proteins, ligands, waters). Requires the gemmi package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gemmi
except ImportError:  # pragma: no cover
    gemmi = None  # type: ignore[assignment]


# Standard amino acid residue names
_AA_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",  # Selenomethionine
}

# Water residue names
_WATER_RESIDUES = {"HOH", "WAT", "H2O"}


@dataclass(frozen=True)
class AtomRecord:
    """A single atom from the mmCIF file."""

    element: str
    atom_name: str
    coords: np.ndarray


@dataclass
class ResidueRecord:
    """A residue from the mmCIF file."""

    chain_id: str
    res_name: str
    res_seq: int
    ins_code: str
    entity_id: Optional[str]
    is_polymer: bool
    is_ligand: bool
    is_water: bool
    atoms: List[AtomRecord]

    @property
    def residue_key(self) -> Tuple[str, int, str, str]:
        """Return or compute residue key."""
        return (self.chain_id, self.res_seq, self.ins_code, self.res_name)


@dataclass(frozen=True)
class SecondarySegment:
    """A secondary structure segment (helix or sheet)."""

    chain_id: str
    start_seq: int
    end_seq: int
    kind: str  # "HELIX" or "SHEET"


@dataclass
class MmcifStructure:
    """Parsed mmCIF structure."""

    residues: List[ResidueRecord]
    secondary_segments: List[SecondarySegment]


class MmcifParseError(ValueError):
    """Raised when mmCIF parsing fails."""

    pass


def _get_mmcif_category(block, name: str) -> Dict[str, List[str]]:
    """Extract a category from an mmCIF block."""
    if gemmi is None:
        raise MmcifParseError("gemmi is required for mmCIF parsing")

    category_name = name if name.startswith("_") else f"_{name}"

    if hasattr(block, "get_mmcif_category"):
        category = block.get_mmcif_category(category_name)
        if category:
            return {k: list(v) for k, v in category.items()}

    if hasattr(block, "find_mmcif_category"):
        table = block.find_mmcif_category(category_name)
        if table:
            tags = [t.split(".", 1)[1] if "." in t else t for t in table.tags]
            values: Dict[str, List[str]] = {tag: [] for tag in tags}
            for row in table:
                for tag in tags:
                    values[tag].append(row[tag])
            return values

    # Fallback: attempt to find a loop by prefix
    loop = block.find_loop(f"{category_name}.")
    if loop is None or not hasattr(loop, "tags"):
        return {}
    tags = [t.split(".", 1)[1] for t in loop.tags]
    loop_values: Dict[str, List[str]] = {tag: [] for tag in tags}
    for row in loop:
        for tag, value in zip(tags, row):
            loop_values[tag].append(value)
    return loop_values


def _as_int(value: str) -> Optional[int]:
    """Convert string to int, returning None for missing values."""
    if value in (".", "?", ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_float(value: str) -> Optional[float]:
    """Convert string to float, returning None for missing values."""
    if value in (".", "?", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_entity_types(block) -> Dict[str, str]:
    """Parse entity types from mmCIF."""
    entity = _get_mmcif_category(block, "entity")
    if not entity:
        return {}
    ids = entity.get("id", [])
    types = entity.get("type", [])
    return {ent_id: ent_type for ent_id, ent_type in zip(ids, types)}


def _parse_secondary_segments(block) -> List[SecondarySegment]:
    """Parse secondary structure segments from mmCIF."""
    segments: List[SecondarySegment] = []

    struct_conf = _get_mmcif_category(block, "struct_conf")
    if struct_conf:
        for chain_id, start, end, conf_type in zip(
            struct_conf.get("beg_label_asym_id", []),
            struct_conf.get("beg_label_seq_id", []),
            struct_conf.get("end_label_seq_id", []),
            struct_conf.get("conf_type_id", []),
        ):
            start_seq = _as_int(start)
            end_seq = _as_int(end)
            if start_seq is None or end_seq is None:
                continue
            kind = "HELIX" if conf_type.upper().startswith("HELX") else "OTHER"
            if kind != "OTHER":
                segments.append(
                    SecondarySegment(
                        chain_id=chain_id,
                        start_seq=start_seq,
                        end_seq=end_seq,
                        kind=kind,
                    )
                )

    struct_sheet_range = _get_mmcif_category(block, "struct_sheet_range")
    if struct_sheet_range:
        for chain_id, start, end in zip(
            struct_sheet_range.get("beg_label_asym_id", []),
            struct_sheet_range.get("beg_label_seq_id", []),
            struct_sheet_range.get("end_label_seq_id", []),
        ):
            start_seq = _as_int(start)
            end_seq = _as_int(end)
            if start_seq is None or end_seq is None:
                continue
            segments.append(
                SecondarySegment(
                    chain_id=chain_id,
                    start_seq=start_seq,
                    end_seq=end_seq,
                    kind="SHEET",
                )
            )

    return segments


def parse_mmcif(
    filepath: str | Path,
    add_hydrogens: bool = False,
) -> MmcifStructure:
    """
    Parse an mmCIF file into an MmcifStructure.

    Parameters
    ----------
    filepath : str or Path
        Path to the mmCIF file.
    add_hydrogens : bool, optional
        If True, add missing hydrogens using pdbfixer/openmm.
        Default is False.

    Returns
    -------
    MmcifStructure
        Parsed structure containing residues and secondary structure info.

    Raises
    ------
    MmcifParseError
        If parsing fails or required dependencies are missing.
    """
    if gemmi is None:
        raise MmcifParseError(
            "gemmi is required for mmCIF parsing. "
            "Install it with: pip install gemmi"
        )

    path = Path(filepath)
    if not path.exists():
        raise MmcifParseError(f"mmCIF file not found: {filepath}")

    if add_hydrogens:
        try:
            from io import StringIO

            from openmm.app import PDBxFile  # type: ignore[import-not-found]
            from pdbfixer import PDBFixer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MmcifParseError(
                "Adding implicit hydrogens requires pdbfixer and openmm. "
                "Install them or rerun without add_hydrogens=True."
            ) from exc

        fixer = PDBFixer(filename=str(path))
        if hasattr(fixer, "findMissingResidues"):
            fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()

        buffer = StringIO()
        PDBxFile.writeFile(fixer.topology, fixer.positions, buffer)
        doc = gemmi.cif.read_string(buffer.getvalue())
        block = doc.sole_block()
    else:
        doc = gemmi.cif.read_file(str(path))
        block = doc.sole_block()

    entity_types = _parse_entity_types(block)
    atom_site = _get_mmcif_category(block, "atom_site")
    if not atom_site:
        raise MmcifParseError("mmCIF is missing atom_site category")

    required = [
        "label_asym_id",
        "label_comp_id",
        "label_seq_id",
        "label_atom_id",
        "type_symbol",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]
    for col in required:
        if col not in atom_site:
            raise MmcifParseError(f"atom_site missing column: {col}")

    residues: Dict[Tuple[str, int, str, str], ResidueRecord] = {}

    n_atoms = len(atom_site["label_asym_id"])
    alt_ids = atom_site.get("label_alt_id", ["."] * n_atoms)
    entity_ids = atom_site.get("label_entity_id", [None] * n_atoms)
    ins_codes = atom_site.get("pdbx_PDB_ins_code", ["."] * n_atoms)

    for i in range(n_atoms):
        alt_id = alt_ids[i] if i < len(alt_ids) else "."
        if alt_id not in (".", "?", "A", "", None, False):
            continue

        chain_id = atom_site["label_asym_id"][i]
        res_name = atom_site["label_comp_id"][i]
        res_seq = _as_int(atom_site["label_seq_id"][i])
        if res_seq is None:
            continue
        atom_name = atom_site["label_atom_id"][i]
        element = atom_site["type_symbol"][i]
        x = _as_float(atom_site["Cartn_x"][i])
        y = _as_float(atom_site["Cartn_y"][i])
        z = _as_float(atom_site["Cartn_z"][i])
        if x is None or y is None or z is None:
            continue

        ins_code = ins_codes[i] if i < len(ins_codes) else "."
        if ins_code in (".", "?"):
            ins_code = ""

        entity_id = entity_ids[i] if i < len(entity_ids) else None
        entity_id_key = entity_id if isinstance(entity_id, str) else ""
        entity_type = entity_types.get(entity_id_key, "")
        is_water = res_name.upper() in _WATER_RESIDUES or entity_type == "water"
        is_polymer = entity_type == "polymer" or res_name.upper() in _AA_RESIDUES
        is_ligand = not is_water and not is_polymer

        key = (chain_id, res_seq, ins_code, res_name)
        if key not in residues:
            residues[key] = ResidueRecord(
                chain_id=chain_id,
                res_name=res_name,
                res_seq=res_seq,
                ins_code=ins_code,
                entity_id=entity_id,
                is_polymer=is_polymer,
                is_ligand=is_ligand,
                is_water=is_water,
                atoms=[],
            )
        residues[key].atoms.append(
            AtomRecord(
                element=element,
                atom_name=atom_name,
                coords=np.array([x, y, z], dtype=float),
            )
        )

    secondary_segments = _parse_secondary_segments(block)

    if not residues:
        raise MmcifParseError("No residues parsed from mmCIF atom_site")

    return MmcifStructure(
        residues=list(residues.values()),
        secondary_segments=secondary_segments,
    )
