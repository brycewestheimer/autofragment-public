# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Machine learning-based fragmentation (future feature)."""
from __future__ import annotations


def check_ml_available() -> None:
    """Check if ML extras are installed."""
    raise NotImplementedError(
        "ML features are planned for a future release. "
        "Stay tuned for autofragment 2.0!"
    )


class MLPartitioner:
    """Future ML-based partitioner."""

    def __init__(self) -> None:
        """Initialize a new MLPartitioner instance."""
        check_ml_available()
