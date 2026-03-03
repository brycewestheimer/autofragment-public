# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Fragment metadata updates."""

from autofragment.core.types import Fragment


def test_fragment_metadata():
    """Test new metadata fields (layer, embedding_type)."""
    f = Fragment(id="F1", layer="QM", embedding_type="electronic")
    assert f.layer == "QM"
    assert f.embedding_type == "electronic"

    # Check to_dict
    d = f.to_dict()
    assert d["layer"] == "QM"
    assert d["embedding_type"] == "electronic"

    # Check from_dict
    f2 = Fragment.from_dict(d)
    assert f2.layer == "QM"
    assert f2.embedding_type == "electronic"

    # Check defaults (backward compatibility)
    f3 = Fragment(id="F2")
    assert f3.layer is None
    assert f3.embedding_type is None
    d3 = f3.to_dict()
    assert "layer" not in d3
    assert "embedding_type" not in d3

    f4 = Fragment.from_dict({"id": "F3"})
    assert f4.layer is None
    assert f4.embedding_type is None
