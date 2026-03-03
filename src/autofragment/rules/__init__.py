# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Fragmentation rules engine.

This module provides a flexible, extensible system for controlling
which bonds can be broken during molecular fragmentation.

Core Classes:
    RuleAction: Enum defining possible actions (MUST_NOT_BREAK, etc.)
    FragmentationRule: Abstract base class for all rules
    RuleEngine: Central engine for applying rules

Pre-defined Rules:
    Common: AromaticRingRule, DoubleBondRule, MetalCoordinationRule
    Biological: PeptideBondRule, DisulfideBondRule, ProlineRingRule
    Materials: SiloxaneBridgeRule, MOFLinkerRule, MetalNodeRule
"""

from .base import BondRule, FragmentationRule, RuleAction, RuleEngine, RuleSet
from .biological import (
    AlphaBetaCarbonRule,
    DisulfideBondRule,
    HydrogenBondRule,
    PeptideBondRule,
    ProlineRingRule,
)
from .common import (
    AromaticRingRule,
    DoubleBondRule,
    FunctionalGroupRule,
    MetalCoordinationRule,
)
from .matsci import (
    MetalNodeRule,
    MOFLinkerRule,
    PerovskiteOctahedralRule,
    SiloxaneBridgeRule,
)

__all__ = [
    # Core
    "RuleAction",
    "FragmentationRule",
    "RuleEngine",
    "RuleSet",
    "BondRule",
    # Common rules
    "AromaticRingRule",
    "DoubleBondRule",
    "MetalCoordinationRule",
    "FunctionalGroupRule",

    # Biological rules
    "PeptideBondRule",
    "DisulfideBondRule",
    "AlphaBetaCarbonRule",
    "ProlineRingRule",
    "HydrogenBondRule",
    # Materials science rules
    "SiloxaneBridgeRule",
    "MOFLinkerRule",
    "MetalNodeRule",
    "PerovskiteOctahedralRule",
]
