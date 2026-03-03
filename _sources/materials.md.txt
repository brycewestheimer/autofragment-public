# Materials Science Fragmentation

AutoFragment provides specialized support for materials science systems, including periodic solids, surfaces, and extended structures like MOFs and zeolites.

## Periodic System Handling

Fundamental to materials science is the treatment of periodic boundary conditions (PBC). AutoFragment handles this through several core components:

### Lattice Vectors
The `Lattice` class represents the crystallographic unit cell. It supports:
*   Conversion between **Fractional** and **Cartesian** coordinates.
*   **Reciprocal Lattice** calculations.
*   Cell volume and parameter extraction.

### Minimum Image Convention
All distance-based calculations (bond detection, radial partitioning) in periodic systems automatically use the **Minimum Image Convention**, ensuring that the shortest distance through periodic boundaries is considered.

### Supercells
AutoFragment can generate supercells (`make_supercell`) that maintain full bonding continuity. This is essential for studying larger domains or performing fragmentation on expanded systems.

## Materials Partitioners

Specialized partitioners are available for materials, inheriting from `MatSciPartitioner`.

*   **RadialPartitioner**: Shell-based fragmentation around a specific center (e.g., an defect site).
*   **SlabPartitioner**: Layer-based fragmentation along a specified lattice axis. High useful for surface science.
*   **UnitCellPartitioner**: Divides a supercell into smaller units based on the underlying grid.
*   **SurfacePartitioner**: Automatically distinguishes between **Top Surface**, **Bottom Surface**, and **Bulk** regions based on vacuum distribution.

## Material-Specific Rules

AutoFragment includes pre-defined rules for common material classes:

### Zeolites
*   **Siloxane bridges** (Si-O-Si) are identified as valid fragmentation points.
*   **Brønsted acid sites** (Al-O(H)-Si) are specifically protected to maintain chemical integrity of the active site.

### Metal-Organic Frameworks (MOFs)
*   **Metal Clusters (Nodes)** are preserved.
*   **Organic Linkers** are kept intact.
*   **Metal-Carboxylate** bonds can be marked as breakable to separate nodes from linkers.

### Polymers
*   **Backbone Detection**: Identifies C-C chains and distinguishes between internal and terminal monomers.
*   **Side-chain Preservation**: Keeps functional side-groups attached to the backbone.

### Perovskites
*   **Octahedral preservation**: Maintains the integrity of $MO_6$ octahedra while allowing for corner-sharing separation.

## Usage Example

```python
from autofragment.partitioners.geometric import SurfacePartitioner
from autofragment.io import read_cif

# Load a slab system
system = read_cif("slab_system.cif")

# Partition by surface vs bulk
partitioner = SurfacePartitioner(surface_axis=2, surface_depth=4.5)
tree = partitioner.partition(system)

# tree.fragments now contains "surface_top", "surface_bottom", and "bulk"
```
