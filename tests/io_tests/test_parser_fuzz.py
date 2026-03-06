# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Fuzz tests for I/O parsers using Hypothesis.

Each test generates random input and asserts the parser either returns a
valid result or raises ValueError/KeyError/TypeError. Any other
exception type (e.g. IndexError, AttributeError) indicates a parser bug
that should be caught and converted to ValueError.
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from autofragment.io.readers.pdb import _parse_atom_record, _parse_conect_record
from autofragment.io.readers.mol2 import _parse_mol2_atom, _parse_mol2_bond
from autofragment.io.readers.sdf import _parse_v2000_atom, _parse_v2000_bond
from autofragment.io.readers.qcschema import qcschema_to_system

_ALLOWED = (ValueError, KeyError, TypeError)


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_atom_record_fuzz(line: str) -> None:
    try:
        result = _parse_atom_record(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_conect_record_fuzz(line: str) -> None:
    try:
        result = _parse_conect_record(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_mol2_atom_fuzz(line: str) -> None:
    try:
        result = _parse_mol2_atom(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_mol2_bond_fuzz(line: str) -> None:
    try:
        result = _parse_mol2_bond(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_v2000_atom_fuzz(line: str) -> None:
    try:
        result = _parse_v2000_atom(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(line=st.text(min_size=0, max_size=200))
def test_parse_v2000_bond_fuzz(line: str) -> None:
    try:
        result = _parse_v2000_bond(line)
    except _ALLOWED:
        pass


@settings(max_examples=200, deadline=None)
@given(
    data=st.fixed_dictionaries(
        {
            "symbols": st.lists(st.text(min_size=1, max_size=3), min_size=0, max_size=5),
            "geometry": st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=20),
        },
        optional={
            "schema_name": st.text(min_size=0, max_size=30),
            "molecular_charge": st.integers(min_value=-10, max_value=10),
            "connectivity": st.lists(
                st.lists(st.integers(min_value=0, max_value=10), min_size=0, max_size=4),
                min_size=0, max_size=5,
            ),
        },
    )
)
def test_qcschema_to_system_fuzz(data: dict) -> None:
    try:
        result = qcschema_to_system(data)
    except _ALLOWED:
        pass
