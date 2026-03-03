# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from autofragment.algorithms.ml_interface import (
    GNNFragmentationModel,
    MLFragmentationModel,
    extract_node_features,
    generate_training_data,
)


def test_cannot_instantiate_abc():
    """Test that MLFragmentationModel cannot be instantiated."""
    with pytest.raises(TypeError):
        MLFragmentationModel()

def test_concrete_implementation():
    """Test that a concrete implementation works."""
    class SimpleModel(MLFragmentationModel):
        def predict(self, system):
            return []

        def fit(self, examples):
            pass

    model = SimpleModel()
    assert model.predict(None) == []
    model.save("test") # should pass

class MockNxGraph:
    def __init__(self, n):
        self.n = n
    def neighbors(self, i):
        return []

class MockMolecularGraphForML:
    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self._graph = MockNxGraph(n_atoms)
        self.atoms = [{"element": "C"}, {"element": "O"}] if n_atoms > 0 else []

    def get_atom(self, idx):
        if idx < len(self.atoms):
            return self.atoms[idx]
        return {"element": "H"}

    def find_rings(self):
        return []

def test_extract_node_features():
    g = MockMolecularGraphForML(2)
    # 2 atoms, features length = 10 (elements) + 4 (deg, chg, ring, aro) = 14
    feats = extract_node_features(g)
    assert feats.shape == (2, 14)
    # Check elements. C is 0th, O is 2nd in our list
    # C
    assert feats[0, 0] == 1.0 # C
    assert feats[0, 2] == 0.0 # O
    # O
    assert feats[1, 0] == 0.0
    assert feats[1, 2] == 1.0

def test_generate_training_data():
    # Placeholder test
    data = generate_training_data([], [])
    assert isinstance(data, list)

def test_gnn_placeholder():
    model = GNNFragmentationModel()
    with pytest.raises(NotImplementedError):
        model.predict(None)
