# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Machine learning interface for fragmentation (Future)."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from autofragment.core.graph import MolecularGraph
    from autofragment.core.types import ChemicalSystem

class MLFragmentationModel(ABC):
    """Abstract interface for ML-based fragmentation.

    Future implementations may include:
    - Graph Neural Networks (GNN)
    - Reinforcement Learning agents
    - Supervised learning from examples
    """

    @abstractmethod
    def predict(
        self,
        system: "ChemicalSystem"
    ) -> List[Tuple[int, int]]:
        """Predict which bonds to break.

        Args:
            system: ChemicalSystem to fragment

        Returns:
            List of bonds to break
        """
        pass

    @abstractmethod
    def fit(
        self,
        examples: List[Tuple["ChemicalSystem", List[Tuple[int, int]]]]
    ) -> None:
        """Train model on example fragmentations.

        Args:
            examples: List of (system, broken_bonds) pairs
        """
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

def extract_node_features(graph: "MolecularGraph") -> np.ndarray:
    """Extract per-node features for ML.

    Features per atom:
    - One-hot element encoding
    - Degree (number of bonds)
    - Formal charge
    - In ring (boolean)
    - Aromatic (boolean)

    Args:
        graph: MolecularGraph

    Returns:
        Array of shape (n_atoms, n_features)
    """
    if graph.n_atoms == 0:
        return np.array([])

    features = []
    # Use graph.get_atom calls.
    # Note: element_one_hot is not defined yet, we'll use a simple version or placeholder

    # Common organic elements
    elements = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']

    for i in range(graph.n_atoms):
        atom = graph.get_atom(i)

        # One-hot
        elem = atom["element"]
        one_hot = [1.0 if elem == e else 0.0 for e in elements]
        if elem not in elements:
             # Basic "other" bin? Or just all zeros
             pass

        # Other features
        degree = len(graph.neighbors(i))
        charge = atom.get("charge", 0.0)

        # Ring/Aromatic check (might be slow if repeated, but this is feature extraction)
        # Assuming is_in_ring and is_aromatic exist or similar
        # is_in_ring is broken out in task spec as "graph.is_in_ring"
        # We need to check if MolecularGraph has these methods or we implement here
        # MolecularGraph has is_in_ring(u, v) for bonds, not atoms.
        # But we can check if atom is in any cycle.
        in_ring = 1.0 if _atom_in_ring(graph, i) else 0.0

        # Aromatic? Need attribute or ring checking
        is_aromatic = 1.0 if atom.get("is_aromatic") else 0.0

        feat = one_hot + [float(degree), float(charge), in_ring, is_aromatic]
        features.append(feat)

    return np.array(features)

def _atom_in_ring(graph: "MolecularGraph", atom_idx: int) -> bool:
    """Check if atom is in any ring."""
    # This involves finding cycles.
    # graph.find_rings() returns basis.
    # If atom is in any basis cycle, it's in a ring.
    # Note: optimization needed for large graphs (cache rings)
    # Since this is extraction, we might just call find_rings once?
    # But graph doesn't cache.
    # For now, simplistic.
    try:
        rings = graph.find_rings()
        for ring in rings:
            if atom_idx in ring:
                return True
    except Exception:
        pass
    return False

def generate_training_data(
    systems: List["ChemicalSystem"],
    partitions: List[List[List[int]]],
    augment: bool = False
) -> List[Tuple["ChemicalSystem", List[Tuple[int, int]]]]:
    """Generate training data from expert fragmentations.

    Args:
        systems: List of chemical systems
        partitions: Corresponding partitions (list of list of indices)
        augment: Apply data augmentation (rotation, permutation - placeholder)

    Returns:
        List of (system, broken_bonds) training examples
    """
    examples: List[Tuple["ChemicalSystem", List[Tuple[int, int]]]] = []
    return examples # Placeholder as per task spec "Pass"

class GNNFragmentationModel(MLFragmentationModel):
    """Graph Neural Network for fragmentation prediction.

    FUTURE: This is a placeholder for GNN implementation.

    Possible architectures:
    - Message Passing Neural Network (MPNN)
    - Graph Attention Network (GAT)
    - SchNet-style continuous-filter convolution

    Training objective: Predict bond break probability
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        """Initialize a new GNNFragmentationModel instance."""
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self._model = None  # Placeholder

    def predict(self, system: "ChemicalSystem") -> List[Tuple[int, int]]:
        """Predict fragment labels for the provided feature matrix."""
        raise NotImplementedError("GNN model not yet implemented")

    def fit(self, examples) -> None:
        """Fit the model using training features and labels."""
        raise NotImplementedError("GNN training not yet implemented")
