# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Feature detection helpers for optional extras."""
from __future__ import annotations

from typing import Dict, Iterable

from autofragment.optional import is_dependency_available

_FEATURE_DEPENDENCIES: Dict[str, Iterable[str]] = {
    "core": (),
    "balanced": ("k_means_constrained",),
    "bio": ("gemmi",),
    "graph": ("networkx", "scipy"),
    "matsci": ("ase", "pymatgen"),
    "docs": (
        "sphinx",
        "myst_parser",
        "sphinx_rtd_theme",
        "breathe",
        "sphinx_autodoc_typehints",
        "sphinx_copybutton",
    ),
    "dev": ("pytest", "pytest_cov", "ruff", "mypy", "pre_commit"),
}


def has_feature(name: str) -> bool:
    """Return True if the optional feature appears to be available."""
    normalized = name.lower()
    if normalized == "core":
        return True

    dependencies = _FEATURE_DEPENDENCIES.get(normalized)
    if dependencies is None:
        raise ValueError(f"Unknown feature '{name}'.")
    if not dependencies:
        return False

    return all(is_dependency_available(module) for module in dependencies)


def list_features() -> Dict[str, bool]:
    """Return a mapping of known features to availability."""
    return {feature: has_feature(feature) for feature in _FEATURE_DEPENDENCIES}
