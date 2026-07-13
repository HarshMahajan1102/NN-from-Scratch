import numpy as np
from src.optimizers.sgd import SGD
from src.optimizers.adam import Adam

class DummyLayer:

    def __init__(self):
        self.weight = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.biases = np.array([[0.5, 1.0]])

        self.dw = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.db = np.array([[0.05, 0.10]])

def test_sgd_updates_weights():

    l = DummyLayer()
    opt = SGD(lr=0.1)

    w = l.weight.copy()
    opt.update(l)
    assert not np.array_equal(l.weight, w)

def test_adam_updates_weights_and_tracks_iteration():

    l = DummyLayer()
    opt = Adam()
    opt.update(l)

    assert opt.it == 1
    assert hasattr(l, "w_momentum")
    assert hasattr(l, "w_cache")
