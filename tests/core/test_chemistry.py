# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for chemistry utilities."""

import pytest

from autofragment.core.chemistry import (
    PERIODIC_TABLE,
    check_valence_satisfied,
    estimate_formal_charge,
    get_element_property,
    infer_bond_order,
)


def test_check_valence_satisfied():
    """Test valence satisfaction logic."""
    # Carbon needs 4
    assert check_valence_satisfied("C", 4.0)
    assert not check_valence_satisfied("C", 3.0)
    assert not check_valence_satisfied("C", 5.0)

    # Hydrogen needs 1
    assert check_valence_satisfied("H", 1.0)
    assert not check_valence_satisfied("H", 0.0)

    # Oxygen needs 2
    assert check_valence_satisfied("O", 2.0)

    # Test tolerance
    assert check_valence_satisfied("C", 3.99)
    assert check_valence_satisfied("C", 4.01)

def test_check_valence_unknown_element():
    """Test valence check with unknown element."""
    assert not check_valence_satisfied("Xz", 1.0)

def test_infer_bond_order():
    """Test bond order inference based on distances."""
    # C-C single (e.g. ethane, ~1.54 A)
    assert infer_bond_order("C", "C", 1.54) == 1.0

    # C=C double (e.g. ethene, ~1.34 A)
    # Sum radii = 0.76+0.76 = 1.52. 1.34 is ~0.88 * 1.52
    assert infer_bond_order("C", "C", 1.34) == 2.0

    # C-C aromatic (e.g. benzene, ~1.40 A)
    # Ratio 1.40/1.52 = 0.92, which is < 0.96 but >= 0.90
    assert infer_bond_order("C", "C", 1.40) == 1.5

    # C#C triple (e.g. ethyne, ~1.20 A)
    # 1.20 is ~0.79 * 1.52
    assert infer_bond_order("C", "C", 1.20) == 3.0

    # C-H single (e.g. ~1.09 A)
    # Sum = 0.76 + 0.31 = 1.07. 1.09 is slightly larger.
    # 1.09 / 1.07 = 1.018 >= 0.96 -> Single
    assert infer_bond_order("C", "H", 1.09) == 1.0

    # C=O double (e.g. ~1.20 - 1.22 A)
    # Sum = 0.76 + 0.66 = 1.42. 1.21 is ~0.85
    assert infer_bond_order("C", "O", 1.21) == 2.0

    # Too far
    assert infer_bond_order("C", "C", 3.0) == 0.0

    # Unknown element -> 1.0 default
    assert infer_bond_order("Xz", "C", 1.5) == 1.0

def test_is_aromatic_ring():
    """Test aromaticity detection heuristic."""
    from autofragment.core.chemistry import is_aromatic_ring

    # Benzene (delocalized) -> all 1.5
    assert is_aromatic_ring([1.5] * 6)

    # Benzene (Kekule) -> alternating 1, 2
    kekule = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert is_aromatic_ring(kekule)

    # Cyclobutadiene (Kekule) -> 1, 2, 1, 2
    assert not is_aromatic_ring([1.0, 2.0, 1.0, 2.0])

    # Cyclooctatetraene (Kekule) -> 4 double bonds (even)
    cot = [1.0, 2.0] * 4
    assert not is_aromatic_ring(cot)

    # Saturated ring
    assert not is_aromatic_ring([1.0] * 6)


def test_estimate_formal_charge():
    """Test formal charge estimation from connectivity."""
    # Methane C (4 bonds). Gr 4. Charge = 4 - 8 + 4 = 0.
    assert estimate_formal_charge("C", 4.0) == 0.0

    # Ammonium N (4 bonds). Gr 5. Charge = 5 - 8 + 4 = +1.
    assert estimate_formal_charge("N", 4.0) == 1.0

    # Ammonia N (3 bonds). Gr 5. Charge = 5 - 8 + 3 = 0.
    assert estimate_formal_charge("N", 3.0) == 0.0

    # Water O (2 bonds). Gr 6. Charge = 6 - 8 + 2 = 0.
    assert estimate_formal_charge("O", 2.0) == 0.0

    # Hydronium O (3 bonds). Gr 6. Charge = 6 - 8 + 3 = +1.
    assert estimate_formal_charge("O", 3.0) == 1.0

    # Hydroxide O (1 bond). Gr 6. Charge = 6 - 8 + 1 = -1.
    assert estimate_formal_charge("O", 1.0) == -1.0

    # Fluoride F (0 bonds). Gr 7. Charge = 7 - 8 + 0 = -1.
    assert estimate_formal_charge("F", 0.0) == -1.0

    # Proton H (0 bonds).
    assert estimate_formal_charge("H", 0.0) == 1.0

    # Borohydride B (4 bonds). Gr 3. Exceptions logic.
    # 4 > 3 -> Charge = 3 - 8 + 4 = -1.
    assert estimate_formal_charge("B", 4.0) == -1.0

    # Borane BH3 (3 bonds). Gr 3. 3 == 3 -> 0.0
    assert estimate_formal_charge("B", 3.0) == 0.0

    # Unknown/Inert? Ne (0 bonds). Gr 8. 8 - 8 + 0 = 0.
    assert estimate_formal_charge("Ne", 0.0) == 0.0




def test_get_element_property_valid():
    """Test retrieving valid properties."""
    assert get_element_property("C", "mass") == 12.011
    assert get_element_property("H", "atomic_number") == 1
    assert get_element_property("O", "electronegativity") == 3.44
    assert get_element_property("N", "valence") == 3

def test_get_element_property_invalid_element():
    """Test retrieving property for invalid element."""
    with pytest.raises(ValueError, match="Unknown element"):
        get_element_property("Xz", "mass")

def test_get_element_property_invalid_property():
    """Test retrieving invalid property."""
    with pytest.raises(ValueError, match="Unknown property"):
        get_element_property("C", "color")

def test_periodic_table_completeness():
    """Test that all required elements (1-54) are present."""
    # Check a few random ones to ensure range is covered
    assert "H" in PERIODIC_TABLE
    assert "He" in PERIODIC_TABLE
    assert "Xe" in PERIODIC_TABLE
    assert "Fe" in PERIODIC_TABLE
    assert "Br" in PERIODIC_TABLE

    # Check that we have 54 elements
    # Actually wait, I only added selected elements in the implementation?
    # No, I should have added all 1-54. Let me double check the implementation.
    # The requirement was "All elements through Xe (1-54) included".
    # I wrote a long list, let's count them or check for specific ones.

    expected_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                        "Ga", "Ge", "As", "Se", "Br", "Kr",
                        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                        "In", "Sn", "Sb", "Te", "I", "Xe"]

    for el in expected_elements:
        assert el in PERIODIC_TABLE, f"Missing element {el}"

def test_data_types():
    """Test data types of properties."""
    c_data = PERIODIC_TABLE["C"]
    assert isinstance(c_data["atomic_number"], int)
    assert isinstance(c_data["mass"], float)
    assert isinstance(c_data["electronegativity"], float)
    assert isinstance(c_data["valence"], int)
