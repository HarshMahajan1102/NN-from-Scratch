import numpy as np
from src.layers.dense import Dense

def test_forward_shape():

    layer = Dense(5, 6)
    X = np.random.randn(4, 5)
    output = layer.forward(X)
    assert output.shape == (4, 6)

def test_backward_shapes():

    layer = Dense(5, 6)
    X = np.random.randn(4, 5)
    output = layer.forward(X)
    dval = np.random.randn(*output.shape)
    layer.backward(dval)

    assert layer.dw.shape == layer.weight.shape
    assert layer.db.shape == layer.biases.shape
    assert layer.di.shape == X.shape

def test_he_initialization_scale():

    n_inputs = 1000
    layer = Dense(n_inputs, 10, init="he")
    expected_std = np.sqrt(2 / n_inputs)
    assert np.isclose(layer.weight.std(), expected_std, atol=0.05)
