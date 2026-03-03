# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration and fixtures for autofragment tests."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def water16_path(fixtures_dir):
    """Path to water16.xyz test file."""
    return fixtures_dir / "water16.xyz"
