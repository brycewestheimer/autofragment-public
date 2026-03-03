# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Example: geometric tiling partitioning."""
from __future__ import annotations

from pathlib import Path

from autofragment.core.types import Atom, ChemicalSystem
from autofragment.partitioners.geometric import TilingPartitioner


def main() -> None:
    atoms = [
        Atom("He", [0.5, 0.5, 0.5]),
        Atom("He", [1.5, 0.5, 0.5]),
        Atom("He", [0.5, 1.5, 0.5]),
        Atom("He", [1.5, 1.5, 0.5]),
        Atom("He", [0.5, 0.5, 1.5]),
        Atom("He", [1.5, 0.5, 1.5]),
        Atom("He", [0.5, 1.5, 1.5]),
        Atom("He", [1.5, 1.5, 1.5]),
    ]
    system = ChemicalSystem(atoms=atoms)

    partitioner = TilingPartitioner(
        tiling_shape="cube",
        n_fragments=8,
        scale=1.0,
        bounds_strategy="atoms",
    )
    tree = partitioner.partition(system)

    output_path = Path("geometric_tiling_output.json")
    tree.to_json(output_path)
    print(f"Wrote {output_path} with {len(tree.fragments)} fragments.")


if __name__ == "__main__":
    main()
