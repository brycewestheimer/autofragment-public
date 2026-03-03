# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for optional dependencies."""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


def is_dependency_available(module: str) -> bool:
    """Return True if the dependency can be imported."""
    return find_spec(module) is not None


def require_dependency(module: str, extra: str, purpose: str) -> ModuleType:
    """Import an optional dependency or raise a helpful error."""
    if not is_dependency_available(module):
        raise ImportError(
            f"{purpose} requires the optional dependency '{module}'. "
            f"Install it with: pip install autofragment[{extra}]"
        )
    return import_module(module)
