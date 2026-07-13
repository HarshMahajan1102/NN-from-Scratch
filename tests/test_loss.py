import numpy as np
from src.losses.cce import CCE
from src.losses.softmax_cce import SoftmaxCCE

def test_cce_zero_for_perfect_prediction():
    loss = CCE()

    y_p = np.array([[1.0, 0.0, 0.0]])
    y = np.array([[1, 0, 0]])

    out = loss.forward(y_p, y)

    assert np.isclose(out, 0.0, atol=1e-6)

def test_softmax_cce_gradient_shape():
    loss = SoftmaxCCE()

    X = np.random.randn(4, 3)
    y = np.array([0, 1, 2, 0])

    loss.forward(X, y)
    loss.backward(loss.output, y)

    assert loss.di.shape == X.shape
