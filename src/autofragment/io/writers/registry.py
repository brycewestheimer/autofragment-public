# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Format registry for extensible I/O.

This module provides a registry system for file format readers and writers,
allowing users to register custom formats and auto-detect file types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from autofragment.core.types import ChemicalSystem, Fragment

# Type aliases for reader/writer functions
ReaderFunc = Callable[[Union[str, Path]], ChemicalSystem]
WriterFunc = Callable[..., None]


class FormatRegistry:
    """
    Registry for file format readers and writers.

    This class provides a central registry for all I/O formats, allowing:
    - Registration of custom readers and writers
    - Auto-detection of file format from extension
    - Listing of supported formats
    - Unified read/write interface

    Examples
    --------
    Register a custom reader:

    >>> def my_reader(filepath):
    ...     # Custom parsing logic
    ...     return ChemicalSystem(...)
    >>> FormatRegistry.register_reader("myformat", my_reader, extensions=[".mfmt"])

    Read a file with auto-detection:

    >>> system = FormatRegistry.read("molecule.pdb")

    List supported formats:

    >>> FormatRegistry.supported_formats()
    {'read': ['pdb', 'mol2', 'sdf', ...], 'write': ['gamess', 'psi4', ...]}
    """

    _readers: Dict[str, ReaderFunc] = {}
    _writers: Dict[str, WriterFunc] = {}
    _extensions: Dict[str, str] = {}  # Maps extension to format name (read)
    _write_extensions: Dict[str, str] = {}  # Maps extension to format name (write)
    _descriptions: Dict[str, str] = {}  # Format descriptions

    @classmethod
    def register_reader(
        cls,
        format_name: str,
        reader: ReaderFunc,
        extensions: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a reader function for a format.

        Parameters
        ----------
        format_name : str
            Name of the format (e.g., "pdb", "mol2").
        reader : callable
            Function that takes filepath and returns ChemicalSystem.
        extensions : list, optional
            File extensions associated with this format (e.g., [".pdb", ".ent"]).
        description : str, optional
            Human-readable description of the format.
        """
        cls._readers[format_name.lower()] = reader

        if extensions:
            for ext in extensions:
                ext_lower = ext.lower()
                if not ext_lower.startswith("."):
                    ext_lower = "." + ext_lower
                cls._extensions[ext_lower] = format_name.lower()

        if description:
            cls._descriptions[format_name.lower()] = description

    @classmethod
    def register_writer(
        cls,
        format_name: str,
        writer: WriterFunc,
        extensions: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a writer function for a format.

        Parameters
        ----------
        format_name : str
            Name of the format.
        writer : callable
            Function that takes (fragments, filepath) and writes the file.
        extensions : list, optional
            File extensions for this format.
        description : str, optional
            Human-readable description.
        """
        cls._writers[format_name.lower()] = writer

        if extensions:
            for ext in extensions:
                ext_lower = ext.lower()
                if not ext_lower.startswith("."):
                    ext_lower = "." + ext_lower
                cls._write_extensions[ext_lower] = format_name.lower()
                # Also add to shared map if not already mapped by a reader
                if ext_lower not in cls._extensions:
                    cls._extensions[ext_lower] = format_name.lower()

        if description:
            cls._descriptions[format_name.lower()] = description

    @classmethod
    def unregister_reader(cls, format_name: str) -> None:
        """Remove a registered reader."""
        format_lower = format_name.lower()
        if format_lower in cls._readers:
            del cls._readers[format_lower]

    @classmethod
    def unregister_writer(cls, format_name: str) -> None:
        """Remove a registered writer."""
        format_lower = format_name.lower()
        if format_lower in cls._writers:
            del cls._writers[format_lower]

    @classmethod
    def _detect_format(cls, filepath: Union[str, Path]) -> Optional[str]:
        """
        Detect format from file extension.

        Parameters
        ----------
        filepath : str or Path
            File path to detect format from.

        Returns
        -------
        str or None
            Detected format name or None if unknown.
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        return cls._extensions.get(ext)

    @classmethod
    def _detect_write_format(cls, filepath: Union[str, Path]) -> Optional[str]:
        """
        Detect write format from file extension.

        Uses the write-specific extension map so that writer extensions
        are not shadowed by reader registrations for the same extension.

        Parameters
        ----------
        filepath : str or Path
            File path to detect format from.

        Returns
        -------
        str or None
            Detected format name or None if unknown.
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        return cls._write_extensions.get(ext)

    @classmethod
    def read(
        cls,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> ChemicalSystem:
        """
        Read file, auto-detecting format if not specified.

        Parameters
        ----------
        filepath : str or Path
            Path to file to read.
        format : str, optional
            Format name. If None, auto-detects from extension.
        **kwargs
            Additional arguments passed to the reader function.

        Returns
        -------
        ChemicalSystem
            Parsed chemical system.

        Raises
        ------
        ValueError
            If format is unknown or cannot be detected.
        """
        if format is None:
            format = cls._detect_format(filepath)

        if format is None:
            raise ValueError(
                f"Cannot detect format for {filepath}. "
                f"Supported extensions: {list(cls._extensions.keys())}"
            )

        format_lower = format.lower()
        if format_lower not in cls._readers:
            raise ValueError(
                f"Unknown format: {format}. "
                f"Supported formats: {list(cls._readers.keys())}"
            )

        reader = cls._readers[format_lower]
        return reader(filepath, **kwargs)

    @classmethod
    def write(
        cls,
        filepath: Union[str, Path],
        fragments: List[Fragment],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write file in specified format.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        fragments : list
            List of Fragment objects to write.
        format : str, optional
            Format name. If None, auto-detects from extension.
        **kwargs
            Additional arguments passed to the writer function.

        Raises
        ------
        ValueError
            If format is unknown.
        """
        if format is None:
            format = cls._detect_write_format(filepath)

        if format is None:
            raise ValueError(
                f"Cannot detect write format for {filepath}. "
                f"Please specify format explicitly."
            )

        format_lower = format.lower()
        if format_lower not in cls._writers:
            raise ValueError(
                f"Unknown write format: {format}. "
                f"Supported formats: {list(cls._writers.keys())}"
            )

        writer = cls._writers[format_lower]
        writer(fragments, filepath, **kwargs)

    @classmethod
    def supported_formats(cls) -> Dict[str, List[str]]:
        """
        Return supported read and write formats.

        Returns
        -------
        dict
            Dictionary with 'read' and 'write' keys containing format lists.
        """
        return {
            "read": sorted(cls._readers.keys()),
            "write": sorted(cls._writers.keys()),
        }

    @classmethod
    def get_format_info(cls, format_name: str) -> Dict[str, Any]:
        """
        Get information about a format.

        Parameters
        ----------
        format_name : str
            Name of the format.

        Returns
        -------
        dict
            Format information including can_read, can_write, description.
        """
        format_lower = format_name.lower()

        # Find associated extensions
        extensions = [
            ext for ext, fmt in cls._extensions.items()
            if fmt == format_lower
        ]

        return {
            "name": format_lower,
            "can_read": format_lower in cls._readers,
            "can_write": format_lower in cls._writers,
            "extensions": extensions,
            "description": cls._descriptions.get(format_lower, ""),
        }

    @classmethod
    def list_formats(cls) -> List[Dict[str, Any]]:
        """
        List all registered formats with their info.

        Returns
        -------
        list
            List of format info dictionaries.
        """
        all_formats = set(cls._readers.keys()) | set(cls._writers.keys())
        return [cls.get_format_info(fmt) for fmt in sorted(all_formats)]


def _register_builtins() -> None:
    """Register all built-in readers and writers."""
    # Readers
    from autofragment.io.readers import (
        read_cif,
        read_gamess_input,
        read_mol2,
        read_nwchem_input,
        read_orca_input,
        read_pdb,
        read_poscar,
        read_psi4_input,
        read_qchem_input,
        read_qcschema,
        read_sdf,
    )

    FormatRegistry.register_reader(
        "pdb", read_pdb,
        extensions=[".pdb", ".ent"],
        description="Protein Data Bank format"
    )
    FormatRegistry.register_reader(
        "mol2", read_mol2,
        extensions=[".mol2"],
        description="Tripos MOL2 format"
    )
    FormatRegistry.register_reader(
        "sdf", read_sdf,
        extensions=[".sdf", ".mol"],
        description="MDL Structure-Data File (V2000/V3000)"
    )
    FormatRegistry.register_reader(
        "qcschema", read_qcschema,
        extensions=[".json", ".qcschema"],
        description="MolSSI QCSchema JSON format"
    )
    FormatRegistry.register_reader(
        "gamess", read_gamess_input,
        extensions=[".inp"],
        description="GAMESS input file"
    )
    FormatRegistry.register_reader(
        "psi4", read_psi4_input,
        extensions=[".dat"],
        description="Psi4 input file"
    )
    FormatRegistry.register_reader(
        "qchem", read_qchem_input,
        extensions=[".in", ".qcin"],
        description="Q-Chem input file"
    )
    FormatRegistry.register_reader(
        "orca", read_orca_input,
        extensions=[".orca"],
        description="ORCA input file"
    )
    FormatRegistry.register_reader(
        "nwchem", read_nwchem_input,
        extensions=[".nw"],
        description="NWChem input file"
    )
    FormatRegistry.register_reader(
        "poscar", read_poscar,
        extensions=[".poscar", ".contcar", ".vasp"],
        description="VASP POSCAR/CONTCAR format"
    )
    FormatRegistry.register_reader(
        "cif", read_cif,
        extensions=[".cif", ".mmcif"],
        description="Crystallographic Information File"
    )

    # Writers – import from individual modules to avoid circular import
    # through autofragment.io.writers.__init__ (which imports this registry).
    from autofragment.io.writers.cfour import write_cfour_fragment
    from autofragment.io.writers.gamess import write_gamess_fmo
    from autofragment.io.writers.molpro import write_molpro_sapt
    from autofragment.io.writers.nwchem import write_nwchem_fragment
    from autofragment.io.writers.orca import write_orca_fragment
    from autofragment.io.writers.pdb_writer import write_pdb_fragments
    from autofragment.io.writers.psi4 import write_psi4_sapt
    from autofragment.io.writers.qchem import write_qchem_efp
    from autofragment.io.writers.qcschema_writer import write_qcschema
    from autofragment.io.writers.turbomole import write_turbomole_fragment
    from autofragment.io.writers.xyz_writer import write_xyz_fragments

    FormatRegistry.register_writer(
        "gamess_fmo", write_gamess_fmo,
        extensions=[".inp"],
        description="GAMESS FMO input file"
    )
    FormatRegistry.register_writer(
        "psi4_sapt", write_psi4_sapt,
        extensions=[".dat"],
        description="Psi4 SAPT input file"
    )
    FormatRegistry.register_writer(
        "qchem_efp", write_qchem_efp,
        extensions=[".in"],
        description="Q-Chem EFP input file"
    )
    FormatRegistry.register_writer(
        "nwchem", write_nwchem_fragment,
        extensions=[".nw"],
        description="NWChem input file"
    )
    FormatRegistry.register_writer(
        "orca", write_orca_fragment,
        extensions=[".orca"],
        description="ORCA input file"
    )
    def _write_qcschema_adapter(fragments, filepath, **kwargs):
        """Adapter for write_qcschema which expects (system, filepath)."""
        from autofragment.core.types import ChemicalSystem

        if isinstance(fragments, ChemicalSystem):
            write_qcschema(fragments, filepath, **kwargs)
        else:
            # Build a ChemicalSystem from fragments and pass fragments for annotation
            all_atoms = []
            for frag in fragments:
                all_atoms.extend(frag.atoms)
            system = ChemicalSystem(atoms=all_atoms, bonds=[], metadata={})
            write_qcschema(system, filepath, fragments=fragments, **kwargs)

    FormatRegistry.register_writer(
        "qcschema", _write_qcschema_adapter,
        extensions=[".json"],
        description="QCSchema JSON output"
    )
    FormatRegistry.register_writer(
        "molpro", write_molpro_sapt,
        extensions=[".com"],
        description="Molpro input file"
    )
    FormatRegistry.register_writer(
        "turbomole", write_turbomole_fragment,
        description="Turbomole coord file"
    )
    FormatRegistry.register_writer(
        "cfour", write_cfour_fragment,
        description="CFOUR ZMAT file"
    )
    FormatRegistry.register_writer(
        "xyz_fragments", write_xyz_fragments,
        extensions=[".xyz"],
        description="XYZ with fragment markers"
    )
    FormatRegistry.register_writer(
        "pdb_fragments", write_pdb_fragments,
        extensions=[".pdb"],
        description="PDB with fragments as chains"
    )


# Auto-register built-in formats on import
_register_builtins()
