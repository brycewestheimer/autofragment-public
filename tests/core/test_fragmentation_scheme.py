# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FragmentationScheme."""

from autofragment.core.types import FragmentationScheme


def test_fragmentation_scheme():
    """Test creation and serialization."""
    scheme = FragmentationScheme(
        algorithm="RMF",
        parameters={"cut_type": "residue", "max_size": 20},
        description="Residue-based Manual Fragmentation"
    )

    assert scheme.algorithm == "RMF"
    assert scheme.parameters["cut_type"] == "residue"
    assert scheme.description.startswith("Residue-based")

    d = scheme.to_dict()
    assert d["algorithm"] == "RMF"

    scheme2 = FragmentationScheme.from_dict(d)
    assert scheme2.algorithm == "RMF"
    assert scheme2.parameters == scheme.parameters
    assert scheme2.description == scheme.description
