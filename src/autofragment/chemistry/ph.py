# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""pH-dependent charge calculation utilities.

This module provides Henderson-Hasselbalch equation implementations
for calculating ionization states at various pH values.
"""

from __future__ import annotations

from typing import List, Optional

from autofragment.data.pka_values import PKA_TERMINAL, PKA_VALUES


def henderson_hasselbalch_acidic(ph: float, pka: float) -> float:
    """Calculate charge contribution for an acidic group.

    For acidic groups: HA ⇌ H+ + A-

    The fraction deprotonated (negative charge) is calculated using:
    fraction_deprotonated = 1 / (1 + 10^(pKa - pH))

    Parameters
    ----------
    ph : float
        pH of the solution
    pka : float
        pKa of the acidic group

    Returns
    -------
    float
        Charge contribution (-1 to 0)
        More negative at higher pH (more deprotonated)

    Examples
    --------
    >>> henderson_hasselbalch_acidic(7.4, 3.65)  # ASP, fully deprotonated
    -0.999...
    >>> henderson_hasselbalch_acidic(3.65, 3.65)  # At pKa, 50% deprotonated
    -0.5
    """
    fraction_deprotonated = 1.0 / (1.0 + 10 ** (pka - ph))
    return -fraction_deprotonated


def henderson_hasselbalch_basic(ph: float, pka: float) -> float:
    """Calculate charge contribution for a basic group.

    For basic groups: BH+ ⇌ B + H+

    The fraction protonated (positive charge) is calculated using:
    fraction_protonated = 1 / (1 + 10^(pH - pKa))

    Parameters
    ----------
    ph : float
        pH of the solution
    pka : float
        pKa of the basic group (conjugate acid)

    Returns
    -------
    float
        Charge contribution (0 to +1)
        More positive at lower pH (more protonated)

    Examples
    --------
    >>> henderson_hasselbalch_basic(7.4, 10.53)  # LYS, mostly protonated
    0.999...
    >>> henderson_hasselbalch_basic(10.53, 10.53)  # At pKa, 50% protonated
    0.5
    """
    fraction_protonated = 1.0 / (1.0 + 10 ** (ph - pka))
    return fraction_protonated


def get_ionization_fraction(ph: float, pka: float, is_basic: bool = False) -> float:
    """Get the ionization fraction for a group.

    Parameters
    ----------
    ph : float
        pH of the solution
    pka : float
        pKa of the group
    is_basic : bool, optional
        If True, treat as basic group (default: acidic)

    Returns
    -------
    float
        Fraction ionized (0 to 1)
    """
    if is_basic:
        return 1.0 / (1.0 + 10 ** (ph - pka))  # Protonated fraction
    else:
        return 1.0 / (1.0 + 10 ** (pka - ph))  # Deprotonated fraction


def get_n_terminus_charge(ph: float, pka: Optional[float] = None) -> float:
    """Get N-terminus charge at a given pH.

    The α-amino group has pKa ~7.7 (average).
    At pH below pKa, it's protonated (NH3+) with charge +1.
    At pH above pKa, it's deprotonated (NH2) with charge 0.

    Parameters
    ----------
    ph : float
        pH of the solution
    pka : float, optional
        pKa of N-terminus (default: 7.7)

    Returns
    -------
    float
        Charge contribution (0 to +1)
    """
    if pka is None:
        pka = PKA_TERMINAL.get("N_TERMINUS", 7.7)
    return henderson_hasselbalch_basic(ph, pka)


def get_c_terminus_charge(ph: float, pka: Optional[float] = None) -> float:
    """Get C-terminus charge at a given pH.

    The α-carboxyl group has pKa ~3.3 (average).
    At pH below pKa, it's protonated (COOH) with charge 0.
    At pH above pKa, it's deprotonated (COO-) with charge -1.

    Parameters
    ----------
    ph : float
        pH of the solution
    pka : float, optional
        pKa of C-terminus (default: 3.3)

    Returns
    -------
    float
        Charge contribution (-1 to 0)
    """
    if pka is None:
        pka = PKA_TERMINAL.get("C_TERMINUS", 3.3)
    return henderson_hasselbalch_acidic(ph, pka)


def get_sidechain_charge(residue: str, ph: float) -> float:
    """Get side chain charge for an amino acid at a given pH.

    Only ionizable residues (ASP, GLU, CYS, TYR, HIS, LYS, ARG, SEC)
    have non-zero charges.

    Parameters
    ----------
    residue : str
        3-letter amino acid code
    ph : float
        pH of the solution

    Returns
    -------
    float
        Side chain charge
    """
    residue = residue.strip().upper()

    # Acidic residues
    acidic = {"ASP", "GLU", "CYS", "TYR", "SEC"}
    # Basic residues
    basic = {"HIS", "LYS", "ARG"}

    pka_key = f"{residue}_SIDECHAIN"
    pka = PKA_VALUES.get(pka_key)

    if pka is None:
        return 0.0

    if residue in acidic:
        return henderson_hasselbalch_acidic(ph, pka)
    elif residue in basic:
        return henderson_hasselbalch_basic(ph, pka)

    return 0.0


def calculate_isoelectric_point(
    residues: List[str],
    precision: float = 0.01,
    ph_range: tuple = (0.0, 14.0),
) -> float:
    """Calculate the isoelectric point (pI) of a protein.

    The pI is the pH at which the net charge is zero.
    Uses bisection method to find the pH.

    Parameters
    ----------
    residues : list of str
        List of 3-letter amino acid codes
    precision : float, optional
        Precision of pI calculation (default: 0.01)
    ph_range : tuple, optional
        pH range to search (default: 0-14)

    Returns
    -------
    float
        Isoelectric point
    """
    ph_low, ph_high = ph_range

    def net_charge(ph: float) -> float:
        """Return or compute net charge."""
        charge = 0.0
        # Terminal charges
        charge += get_n_terminus_charge(ph)
        charge += get_c_terminus_charge(ph)
        # Side chain charges
        for res in residues:
            charge += get_sidechain_charge(res, ph)
        return charge

    # Bisection method
    while (ph_high - ph_low) > precision:
        ph_mid = (ph_low + ph_high) / 2
        charge_mid = net_charge(ph_mid)

        if charge_mid > 0:
            ph_low = ph_mid
        else:
            ph_high = ph_mid

    return (ph_low + ph_high) / 2


def validate_ph(ph: float) -> None:
    """Validate pH value is in acceptable range.

    Parameters
    ----------
    ph : float
        pH value to validate

    Raises
    ------
    ValueError
        If pH is outside 0-14 range
    """
    if not 0 <= ph <= 14:
        raise ValueError(f"pH must be between 0 and 14, got {ph}")
