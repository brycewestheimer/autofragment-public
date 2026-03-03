# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Chemistry utilities and periodic table data."""

from typing import Any, Dict, List

# Element data: atomic_number, mass (g/mol), electronegativity (Pauling), common valence
PERIODIC_TABLE: Dict[str, Dict[str, Any]] = {
    "H": {"atomic_number": 1, "mass": 1.008, "electronegativity": 2.20, "valence": 1},
    "He": {"atomic_number": 2, "mass": 4.0026, "electronegativity": 0.0, "valence": 0},
    "Li": {"atomic_number": 3, "mass": 6.94, "electronegativity": 0.98, "valence": 1},
    "Be": {"atomic_number": 4, "mass": 9.0122, "electronegativity": 1.57, "valence": 2},
    "B": {"atomic_number": 5, "mass": 10.81, "electronegativity": 2.04, "valence": 3},
    "C": {"atomic_number": 6, "mass": 12.011, "electronegativity": 2.55, "valence": 4},
    "N": {"atomic_number": 7, "mass": 14.007, "electronegativity": 3.04, "valence": 3},
    "O": {"atomic_number": 8, "mass": 15.999, "electronegativity": 3.44, "valence": 2},
    "F": {"atomic_number": 9, "mass": 18.998, "electronegativity": 3.98, "valence": 1},
    "Ne": {"atomic_number": 10, "mass": 20.180, "electronegativity": 0.0, "valence": 0},
    "Na": {"atomic_number": 11, "mass": 22.990, "electronegativity": 0.93, "valence": 1},
    "Mg": {"atomic_number": 12, "mass": 24.305, "electronegativity": 1.31, "valence": 2},
    "Al": {"atomic_number": 13, "mass": 26.982, "electronegativity": 1.61, "valence": 3},
    "Si": {"atomic_number": 14, "mass": 28.085, "electronegativity": 1.90, "valence": 4},
    "P": {"atomic_number": 15, "mass": 30.974, "electronegativity": 2.19, "valence": 5},
    "S": {"atomic_number": 16, "mass": 32.06, "electronegativity": 2.58, "valence": 6},
    "Cl": {"atomic_number": 17, "mass": 35.45, "electronegativity": 3.16, "valence": 1},
    "Ar": {"atomic_number": 18, "mass": 39.948, "electronegativity": 0.0, "valence": 0},
    "K": {"atomic_number": 19, "mass": 39.098, "electronegativity": 0.82, "valence": 1},
    "Ca": {"atomic_number": 20, "mass": 40.078, "electronegativity": 1.00, "valence": 2},
    "Sc": {"atomic_number": 21, "mass": 44.956, "electronegativity": 1.36, "valence": 3},
    "Ti": {"atomic_number": 22, "mass": 47.867, "electronegativity": 1.54, "valence": 4},
    "V": {"atomic_number": 23, "mass": 50.942, "electronegativity": 1.63, "valence": 5},
    "Cr": {"atomic_number": 24, "mass": 51.996, "electronegativity": 1.66, "valence": 6},
    "Mn": {"atomic_number": 25, "mass": 54.938, "electronegativity": 1.55, "valence": 7},
    "Fe": {"atomic_number": 26, "mass": 55.845, "electronegativity": 1.83, "valence": 3},
    "Co": {"atomic_number": 27, "mass": 58.933, "electronegativity": 1.88, "valence": 3},
    "Ni": {"atomic_number": 28, "mass": 58.693, "electronegativity": 1.91, "valence": 2},
    "Cu": {"atomic_number": 29, "mass": 63.546, "electronegativity": 1.90, "valence": 2},
    "Zn": {"atomic_number": 30, "mass": 65.38, "electronegativity": 1.65, "valence": 2},
    "Ga": {"atomic_number": 31, "mass": 69.723, "electronegativity": 1.81, "valence": 3},
    "Ge": {"atomic_number": 32, "mass": 72.630, "electronegativity": 2.01, "valence": 4},
    "As": {"atomic_number": 33, "mass": 74.922, "electronegativity": 2.18, "valence": 5},
    "Se": {"atomic_number": 34, "mass": 78.971, "electronegativity": 2.55, "valence": 6},
    "Br": {"atomic_number": 35, "mass": 79.904, "electronegativity": 2.96, "valence": 1},
    "Kr": {"atomic_number": 36, "mass": 83.798, "electronegativity": 3.00, "valence": 0},
    "Rb": {"atomic_number": 37, "mass": 85.468, "electronegativity": 0.82, "valence": 1},
    "Sr": {"atomic_number": 38, "mass": 87.62, "electronegativity": 0.95, "valence": 2},
    "Y": {"atomic_number": 39, "mass": 88.906, "electronegativity": 1.22, "valence": 3},
    "Zr": {"atomic_number": 40, "mass": 91.224, "electronegativity": 1.33, "valence": 4},
    "Nb": {"atomic_number": 41, "mass": 92.906, "electronegativity": 1.6, "valence": 5},
    "Mo": {"atomic_number": 42, "mass": 95.95, "electronegativity": 2.16, "valence": 6},
    "Tc": {"atomic_number": 43, "mass": 98.0, "electronegativity": 1.9, "valence": 7},
    "Ru": {"atomic_number": 44, "mass": 101.07, "electronegativity": 2.2, "valence": 8},
    "Rh": {"atomic_number": 45, "mass": 102.91, "electronegativity": 2.28, "valence": 6},
    "Pd": {"atomic_number": 46, "mass": 106.42, "electronegativity": 2.20, "valence": 4},
    "Ag": {"atomic_number": 47, "mass": 107.87, "electronegativity": 1.93, "valence": 1},
    "Cd": {"atomic_number": 48, "mass": 112.41, "electronegativity": 1.69, "valence": 2},
    "In": {"atomic_number": 49, "mass": 114.82, "electronegativity": 1.78, "valence": 3},
    "Sn": {"atomic_number": 50, "mass": 118.71, "electronegativity": 1.96, "valence": 4},
    "Sb": {"atomic_number": 51, "mass": 121.76, "electronegativity": 2.05, "valence": 5},
    "Te": {"atomic_number": 52, "mass": 127.60, "electronegativity": 2.1, "valence": 6},
    "I": {"atomic_number": 53, "mass": 126.90, "electronegativity": 2.66, "valence": 1},
    "Xe": {"atomic_number": 54, "mass": 131.29, "electronegativity": 2.60, "valence": 0},
}

def get_element_property(element: str, prop: str) -> Any:
    """
    Get a property for an element.

    Args:
        element: Chemical symbol (e.g., 'C', 'H')
        prop: Property name ('atomic_number', 'mass', 'electronegativity', 'valence')

    Returns:
        The value of the property.

    Raises:
        ValueError: If element or property is not found.
    """
    if element not in PERIODIC_TABLE:
        raise ValueError(f"Unknown element: {element}")

    element_data = PERIODIC_TABLE[element]
    if prop not in element_data:
        raise ValueError(f"Unknown property: {prop} for element {element}")

    return element_data[prop]

def check_valence_satisfied(element: str, bond_order_sum: float) -> bool:
    """Check if the total bond order satisfies the valence of the element.

    Args:
        element: Chemical symbol (e.g., 'C').
        bond_order_sum: Sum of bond orders connected to the atom.

    Returns:
        True if the bond order sum matches the standard valence, False otherwise.

    Note:
        This is a simplified check and may not cover all hypervalent or
        charged states. It primarily checks against the standard valence.
    """
    try:
        standard_valence = get_element_property(element, "valence")
        # Allow small floating point tolerance
        return abs(standard_valence - bond_order_sum) < 0.1
    except ValueError:
        return False

# Covalent radii in Angstroms (approximate)
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Kr": 1.16, "I": 1.39, "Xe": 1.40
}

def infer_bond_order(element1: str, element2: str, distance: float) -> float:
    """Infer bond order from distance between two atoms.

    Uses covalent radii to estimate bond order.

    Args:
        element1: Symbol of first element.
        element2: Symbol of second element.
        distance: Distance in Angstroms.

    Returns:
        Estimated bond order (1.0, 2.0, or 3.0). Returns 0.0 if not bonded (too far).
    """
    if element1 not in COVALENT_RADII or element2 not in COVALENT_RADII:
        return 1.0 # Default to single if unknown elements

    r1 = COVALENT_RADII[element1]
    r2 = COVALENT_RADII[element2]
    sum_radii = r1 + r2

    ratio = distance / sum_radii

    # We use some generous thresholds to categorize
    if distance > sum_radii + 0.4:
        return 0.0 # Likely not bonded

    # Thresholds based on typical ratios:
    # Triple: < 0.82
    # Double: < 0.90 (e.g. 1.34/1.52 = 0.88)
    # Aromatic: < 0.96 (e.g. 1.40/1.52 = 0.92, 1.44/1.52 = 0.94)
    # Single: >= 0.96 (e.g. 1.54/1.52 = 1.01)

    if ratio < 0.82:
        return 3.0
    elif ratio < 0.90:
        return 2.0
    elif ratio < 0.96:
        return 1.5
    else:
        return 1.0

def is_aromatic_ring(bond_orders: List[float]) -> bool:
    """Check if a ring is aromatic based on bond orders.

    Args:
        bond_orders: List of bond orders in the ring.

    Returns:
        True if the ring appears to be aromatic.
    """
    if not bond_orders:
        return False

    # 1. Check for delocalized bonds (order 1.5)
    # Our infer_bond_order returns 1.5 for 1.39-1.45 A range C-C
    if any(abs(bo - 1.5) < 0.1 for bo in bond_orders):
        return True

    # 2. Check for Kekule structure (alternating single/double)
    # Huckel's rule for annulenes: 4n + 2 pi electrons
    # Each double bond contributes 2 pi electrons
    # So 2 * n_double = 4n + 2 => n_double is odd
    n_double = sum(1 for bo in bond_orders if abs(bo - 2.0) < 0.1)

    # Must have at least one double bond and match parity
    # Also ensures simple cycles like cyclobutadiene (n=2, even) are excluded
    # Note: This simple heuristic fails for heterocycles if they don't have 1.5 bonds,
    # but most aromatic heterocycles have intermediate bond lengths.
    if n_double > 0 and n_double % 2 == 1:
        return True

    return False


def get_valence_electrons(element: str) -> int:
    """Get number of valence electrons (group number) for main group elements."""
    # Simplified lookup for common elements
    groups = {
        "H": 1, "Li": 1, "Na": 1, "K": 1,
        "Be": 2, "Mg": 2, "Ca": 2,
        "B": 3, "Al": 3,
        "C": 4, "Si": 4,
        "N": 5, "P": 5,
        "O": 6, "S": 6,
        "F": 7, "Cl": 7, "Br": 7, "I": 7,
        "He": 8, "Ne": 8, "Ar": 8, "Kr": 8, "Xe": 8
    }
    return groups.get(element, 0)

def estimate_formal_charge(element: str, bond_order_sum: float) -> float:
    """Estimate formal charge assuming octet rule for main group elements.

    Formula used: Charge = Valence_e - 8 + Bonds
    (assuming completed octet by lone pairs)

    Exceptions handled:
    - Hydrogen (Duet rule)
    - Group 1-3 (Electron deficient neutral states)

    Args:
        element: Chemical symbol.
        bond_order_sum: Total bond order.

    Returns:
        Estimated formal charge (e.g., 0.0, 1.0, -1.0).
    """
    valence_e = get_valence_electrons(element)
    if valence_e == 0:
        return 0.0

    # Hydrogen case
    if element == "H":
        if bond_order_sum == 0:
            return 1.0 # H+
        # H with 1 bond -> 0
        return 0.0

    # Electron deficient groups (1-3)
    if valence_e <= 3:
        # If bonds match valence, neutral (e.g. BH3, AlCl3)
        if abs(bond_order_sum - valence_e) < 0.1:
            return 0.0
        # If more bonds? e.g. BH4- (3 - 8 + 4 = -1)
        if bond_order_sum > valence_e:
            return float(valence_e - 8 + bond_order_sum)

    # Main group (C, N, O, F...) - assume octet
    return float(valence_e - 8 + bond_order_sum)



