# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Output formatting for autofragment results.

This module provides functions for writing FragmentTree objects
to JSON files in the autofragment output format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from autofragment.core.types import FragmentTree


def write_json(
    tree: FragmentTree,
    filepath: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Write a FragmentTree to a JSON file.

    This is a convenience wrapper around FragmentTree.to_json().

    Parameters
    ----------
    tree : FragmentTree
        The fragment tree to write.
    filepath : str or Path
        Output file path.
    indent : int, optional
        JSON indentation level. Default is 2.
    """
    tree.to_json(filepath, indent=indent)


def format_partitioning_info(
    algorithm: str,
    n_fragments: int,
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Format partitioning metadata for output.

    Parameters
    ----------
    algorithm : str
        Name of the clustering algorithm used.
    n_fragments : int
        Number of fragments.

    Returns
    -------
    Dict[str, Any]
        Partitioning metadata dictionary (including any extra metadata).
    """
    info = {
        "algorithm": algorithm,
        "n_fragments": n_fragments,
    }
    info.update(metadata)
    return info


def format_source_info(
    filepath: str | Path,
    file_format: str = "xyz",
) -> Dict[str, str]:
    """
    Format source file metadata for output.

    Parameters
    ----------
    filepath : str or Path
        Path to the source file.
    file_format : str, optional
        File format string. Default is "xyz".

    Returns
    -------
    Dict[str, str]
        Source metadata dictionary.
    """
    return {
        "file": str(Path(filepath).name),
        "format": file_format,
    }
