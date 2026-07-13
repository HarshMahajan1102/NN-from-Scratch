import numpy as np
from src.activations.relu import ReLU

def test_forward_clips_negative_values():

    relu = ReLU()
    X = np.array([[-1, 2, -3, 4]])
    output = relu.forward(X)
    assert np.array_equal(output, np.array([[0, 2, 0, 4]]))

def test_backward_zeroes_negative_gradients():

    relu = ReLU()
    X = np.array([[-1, 2, -3, 4]])
    relu.forward(X)
    dval = np.ones_like(X, dtype=float)
    relu.backward(dval)
    assert np.array_equal(relu.di, np.array([[0, 1, 0, 1]]))
