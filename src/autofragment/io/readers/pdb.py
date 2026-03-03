# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
PDB (Protein Data Bank) file reader.

This module provides functions for reading PDB files into ChemicalSystem objects,
including support for ATOM/HETATM records, CONECT bond records, and multi-MODEL files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from autofragment.core.bonds import COVALENT_RADII
from autofragment.core.types import Atom, ChemicalSystem
from autofragment.io.readers.base import ChunkedReader, LazyAtomIterator

# Mapping of common atom name patterns to element symbols
_ELEMENT_PATTERNS = [
    (re.compile(r"^([A-Z][a-z]?)"), None),  # Standard element (Ca, Fe, etc.)
]

# Common 2-character elements in PDB
_TWO_CHAR_ELEMENTS = {
    "CL", "BR", "FE", "MG", "MN", "ZN", "CA", "NA", "CU", "CO", "NI", "SE", "AS"
}


def _infer_element(atom_name: str) -> str:
    """
    Infer element symbol from PDB atom name.

    PDB atom names are left-justified for 2-char elements (e.g., "FE  "),
    and right-justified for 1-char elements (e.g., " CA ").

    Parameters
    ----------
    atom_name : str
        The atom name from the PDB file (columns 13-16).

    Returns
    -------
    str
        Inferred element symbol.
    """
    name = atom_name.strip().upper()

    if not name:
        return "X"  # Unknown

    # Check for 2-character elements first
    if len(name) >= 2 and name[:2] in _TWO_CHAR_ELEMENTS:
        return name[:2].capitalize()

    # Check for single-character elements
    first_char = name[0]
    if first_char.isdigit():
        # Some atom names start with numbers (e.g., "1HG")
        for c in name:
            if c.isalpha():
                return c.upper()
        return "X"

    # Return capitalized first letter
    return first_char.upper()


def _parse_atom_record(line: str) -> Dict[str, Any]:
    """
    Parse ATOM/HETATM record from PDB file.

    PDB format (columns are 1-indexed):
    - 1-6: Record type (ATOM/HETATM)
    - 7-11: Atom serial number
    - 13-16: Atom name
    - 17: Alternate location indicator
    - 18-20: Residue name
    - 22: Chain ID
    - 23-26: Residue sequence number
    - 27: Code for insertions of residues
    - 31-38: X coordinate
    - 39-46: Y coordinate
    - 47-54: Z coordinate
    - 55-60: Occupancy
    - 61-66: Temperature factor
    - 77-78: Element symbol
    - 79-80: Charge

    Parameters
    ----------
    line : str
        A single line from a PDB file starting with ATOM or HETATM.

    Returns
    -------
    dict
        Parsed atom data with keys: serial, name, alt_loc, residue, chain,
        resnum, insert_code, x, y, z, occupancy, temp_factor, element, charge.
    """
    record_type = line[:6].strip()

    # Parse coordinates (required)
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid coordinates in PDB line: {line!r}") from e

    # Parse atom serial (required)
    try:
        serial = int(line[6:11])
    except (ValueError, IndexError):
        serial = 0

    # Parse atom name
    atom_name = line[12:16] if len(line) > 16 else ""

    # Parse element (columns 77-78, or infer from name)
    element = ""
    if len(line) >= 78:
        element = line[76:78].strip()
    if not element:
        element = _infer_element(atom_name)

    # Parse optional fields
    alt_loc = line[16] if len(line) > 16 else ""
    residue = line[17:20].strip() if len(line) > 20 else ""
    chain = line[21] if len(line) > 21 else ""

    try:
        resnum = int(line[22:26])
    except (ValueError, IndexError):
        resnum = 0

    insert_code = line[26] if len(line) > 26 else ""

    try:
        occupancy = float(line[54:60])
    except (ValueError, IndexError):
        occupancy = 1.0

    try:
        temp_factor = float(line[60:66])
    except (ValueError, IndexError):
        temp_factor = 0.0

    # Parse charge (columns 79-80, format like "2+" or "1-")
    charge = 0.0
    if len(line) >= 80:
        charge_str = line[78:80].strip()
        if charge_str:
            try:
                if charge_str.endswith("+"):
                    charge = float(charge_str[:-1] or "1")
                elif charge_str.endswith("-"):
                    charge = -float(charge_str[:-1] or "1")
            except ValueError:
                pass

    return {
        "record_type": record_type,
        "serial": serial,
        "name": atom_name.strip(),
        "alt_loc": alt_loc.strip(),
        "residue": residue,
        "chain": chain.strip(),
        "resnum": resnum,
        "insert_code": insert_code.strip(),
        "x": x,
        "y": y,
        "z": z,
        "occupancy": occupancy,
        "temp_factor": temp_factor,
        "element": element,
        "charge": charge,
    }


class LazyPdbAtomIterator(LazyAtomIterator[Atom]):
    """Lazy iterator for PDB atoms with MODEL/altLoc handling."""

    def __init__(
        self,
        filepath: Union[str, Path],
        model: int = 1,
        alt_loc: Optional[str] = None,
    ) -> None:
        """Initialize a new LazyPdbAtomIterator instance."""
        self.path = Path(filepath)
        self.model = model
        self.alt_loc = alt_loc
        self._current_model = 1
        self._in_target_model = model == 1
        self._seen_alt_locs: Dict[Tuple[str, int, str], str] = {}

        super().__init__(
            self.path,
            self._parse_atom_line,
            record_prefixes=("ATOM", "HETATM", "MODEL", "ENDMDL"),
        )

    def _parse_atom_line(self, line: str) -> Optional[Atom]:
        """Internal helper to parse atom line."""
        record = line[:6].strip()
        if record == "MODEL":
            try:
                self._current_model = int(line[10:14])
            except (ValueError, IndexError):
                self._current_model += 1
            self._in_target_model = self._current_model == self.model
            return None
        if record == "ENDMDL":
            if self._current_model == self.model:
                raise StopIteration
            return None
        if record not in ("ATOM", "HETATM") or not self._in_target_model:
            return None

        atom_data = _parse_atom_record(line)
        atom_alt_loc = atom_data["alt_loc"]

        if self.alt_loc is not None:
            if atom_alt_loc and atom_alt_loc != self.alt_loc:
                return None
        else:
            key = (atom_data["chain"], atom_data["resnum"], atom_data["name"])
            if key in self._seen_alt_locs:
                if atom_alt_loc != self._seen_alt_locs[key]:
                    return None
            elif atom_alt_loc:
                self._seen_alt_locs[key] = atom_alt_loc

        return Atom(
            symbol=atom_data["element"],
            coords=np.array([atom_data["x"], atom_data["y"], atom_data["z"]]),
            charge=atom_data["charge"],
        )

    def __len__(self) -> int:
        """Return the number of contained items."""
        count = 0
        current_model = 1
        in_target = self.model == 1
        seen_alt_locs: Dict[Tuple[str, int, str], str] = {}
        with open(self.path, "r") as handle:
            for line in handle:
                record = line[:6].strip()
                if record == "MODEL":
                    try:
                        current_model = int(line[10:14])
                    except (ValueError, IndexError):
                        current_model += 1
                    in_target = current_model == self.model
                    continue
                if record == "ENDMDL":
                    if current_model == self.model:
                        break
                    continue
                if record not in ("ATOM", "HETATM") or not in_target:
                    continue
                atom_data = _parse_atom_record(line)
                atom_alt_loc = atom_data["alt_loc"]
                if self.alt_loc is not None:
                    if atom_alt_loc and atom_alt_loc != self.alt_loc:
                        continue
                else:
                    key = (atom_data["chain"], atom_data["resnum"], atom_data["name"])
                    if key in seen_alt_locs:
                        if atom_alt_loc != seen_alt_locs[key]:
                            continue
                    elif atom_alt_loc:
                        seen_alt_locs[key] = atom_alt_loc
                count += 1
        return count


def iter_pdb_atoms(
    filepath: Union[str, Path],
    model: int = 1,
    alt_loc: Optional[str] = None,
) -> Iterator[Atom]:
    """Iterate over PDB atoms lazily."""
    return LazyPdbAtomIterator(filepath, model=model, alt_loc=alt_loc)


def read_pdb_lazy(
    filepath: Union[str, Path],
    chunk_size: int = 10000,
    model: int = 1,
    alt_loc: Optional[str] = None,
) -> ChunkedReader:
    """Return a chunked reader for lazy PDB processing."""
    iterator = LazyPdbAtomIterator(filepath, model=model, alt_loc=alt_loc)
    return ChunkedReader(iterator=iterator, chunk_size=chunk_size)


def _parse_conect_record(line: str) -> List[Tuple[int, int]]:
    """
    Parse CONECT record from PDB file.

    CONECT format (columns are 1-indexed):
    - 1-6: CONECT
    - 7-11: Atom serial number
    - 12-16: Serial number of bonded atom 1
    - 17-21: Serial number of bonded atom 2
    - 22-26: Serial number of bonded atom 3
    - 27-31: Serial number of bonded atom 4
    (may repeat for more connections)

    Parameters
    ----------
    line : str
        A CONECT line from a PDB file.

    Returns
    -------
    list
        List of (atom1_serial, atom2_serial) tuples for each bond.
    """
    bonds: List[Tuple[int, int]] = []

    # Parse the source atom
    try:
        source = int(line[6:11])
    except (ValueError, IndexError):
        return bonds

    # Parse bonded atoms (each field is 5 characters)
    for i in range(4):  # Up to 4 bonds per CONECT record
        start = 11 + i * 5
        end = start + 5
        if end > len(line):
            break
        try:
            target = int(line[start:end])
            if target > 0:
                # Only add bond once (when source < target to avoid duplicates)
                if source < target:
                    bonds.append((source, target))
        except ValueError:
            continue

    return bonds


def read_pdb(
    filepath: Union[str, Path],
    parse_conect: bool = True,
    infer_bonds: bool = True,
    model: int = 1,
    alt_loc: Optional[str] = None,
) -> ChemicalSystem:
    """
    Read PDB file into ChemicalSystem.

    Parameters
    ----------
    filepath : str or Path
        Path to PDB file.
    parse_conect : bool, optional
        Parse CONECT records for explicit bonds. Default is True.
    infer_bonds : bool, optional
        Infer bonds from distances if CONECT missing. Default is True.
    model : int, optional
        Which MODEL to read for multi-model files (1-indexed). Default is 1.
    alt_loc : str, optional
        Filter atoms by alternate location indicator. If None, takes the first
        alt location encountered or atoms with no alt location. Default is None.

    Returns
    -------
    ChemicalSystem
        ChemicalSystem with atoms and bonds.

    Examples
    --------
    >>> system = read_pdb("protein.pdb")
    >>> print(f"Read {system.n_atoms} atoms and {system.n_bonds} bonds")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {filepath}")

    atom_records: List[Dict[str, Any]] = []
    conect_bonds: List[Tuple[int, int]] = []
    serial_to_index: Dict[int, int] = {}
    current_model = 1
    in_target_model = (model == 1)  # Start in model 1 if no MODEL record
    seen_alt_locs: Dict[Tuple[str, int, str], str] = {}  # (chain, resnum, name) -> first alt_loc

    with open(path, "r") as f:
        for line in f:
            record = line[:6].strip()

            if record == "MODEL":
                try:
                    current_model = int(line[10:14])
                except (ValueError, IndexError):
                    current_model += 1
                in_target_model = (current_model == model)

            elif record == "ENDMDL":
                if current_model == model:
                    break  # Done with target model

            elif record in ("ATOM", "HETATM") and in_target_model:
                atom_data = _parse_atom_record(line)

                # Handle alternate locations
                atom_alt_loc = atom_data["alt_loc"]
                if alt_loc is not None:
                    # Filter by specific alt_loc
                    if atom_alt_loc and atom_alt_loc != alt_loc:
                        continue
                else:
                    # Take first alt_loc or empty
                    key = (atom_data["chain"], atom_data["resnum"], atom_data["name"])
                    if key in seen_alt_locs:
                        if atom_alt_loc != seen_alt_locs[key]:
                            continue  # Skip alternate conformations
                    elif atom_alt_loc:
                        seen_alt_locs[key] = atom_alt_loc

                # Map serial to index
                serial_to_index[atom_data["serial"]] = len(atom_records)
                atom_records.append(atom_data)

            elif record == "CONECT" and parse_conect:
                conect_pairs = _parse_conect_record(line)
                conect_bonds.extend(conect_pairs)

    # Build atoms list
    atoms = [
        Atom(
            symbol=rec["element"],
            coords=np.array([rec["x"], rec["y"], rec["z"]]),
            charge=rec["charge"],
        )
        for rec in atom_records
    ]

    # Build bonds list
    bonds: List[Dict[str, Any]] = []

    if conect_bonds:
        # Use CONECT records
        for serial1, serial2 in conect_bonds:
            if serial1 in serial_to_index and serial2 in serial_to_index:
                bonds.append({
                    "atom1": serial_to_index[serial1],
                    "atom2": serial_to_index[serial2],
                    "order": 1.0,  # CONECT doesn't specify order
                })
    elif infer_bonds and atoms:
        # Infer bonds from distances
        bonds = _infer_bonds_from_distances(atoms)

    # Build metadata
    metadata: Dict[str, Any] = {
        "source_format": "pdb",
        "source_file": str(path.name),
        "model": model,
    }

    # Add residue info if available
    if atom_records:
        residue_info: List[Dict[str, Any]] = []
        for rec in atom_records:
            residue_info.append({
                "atom_name": rec["name"],
                "residue": rec["residue"],
                "chain": rec["chain"],
                "resnum": rec["resnum"],
            })
        metadata["residue_info"] = residue_info

    return ChemicalSystem(atoms=atoms, bonds=bonds, metadata=metadata)


def _infer_bonds_from_distances(
    atoms: List[Atom],
    tolerance: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Infer bonds between atoms based on covalent radii.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.
    tolerance : float, optional
        Tolerance above sum of covalent radii to consider bonded.
        Default is 0.4 Angstroms.

    Returns
    -------
    list
        List of bond dictionaries.
    """
    bonds = []
    n = len(atoms)

    for i in range(n):
        for j in range(i + 1, n):
            atom_i = atoms[i]
            atom_j = atoms[j]

            # Get covalent radii
            r_i = COVALENT_RADII.get(atom_i.symbol.upper(), 1.5)
            r_j = COVALENT_RADII.get(atom_j.symbol.upper(), 1.5)

            # Calculate distance
            dist = np.linalg.norm(atom_i.coords - atom_j.coords)

            # Check if bonded
            max_dist = r_i + r_j + tolerance
            if dist <= max_dist and dist > 0.4:  # Minimum distance check
                bonds.append({
                    "atom1": i,
                    "atom2": j,
                    "order": 1.0,
                })

    return bonds
