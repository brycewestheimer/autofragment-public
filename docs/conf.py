# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
from datetime import date

# Ensure the src-layout package is importable for autodoc
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


project = "autofragment"
author = "Bryce M. Westheimer"
copyright = f"{date.today().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "breathe",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]


# MyST (Markdown)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "strikethrough",
    "tasklist",
]


autosummary_generate = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


_DOXYGEN_XML = os.path.join(os.path.dirname(__file__), "_doxygen", "xml")
breathe_projects = {"autofragment": _DOXYGEN_XML} if os.path.isdir(_DOXYGEN_XML) else {}
breathe_default_project = "autofragment"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
