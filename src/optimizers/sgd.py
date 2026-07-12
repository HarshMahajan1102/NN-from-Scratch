import numpy as np

class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr 

    def update(self, layer):
        layer.weight -= self.lr * layer.dw
        layer.biases -= self.lr * layer.db


if __name__ == "__main__":

    class DummyLayer:
        def __init__(self):
            self.weight = np.array([
                [1.0, 2.0],
                [3.0, 4.0]
            ])

            self.biases = np.array([
                [0.5, 1.0]
            ])

            self.dw = np.array([
                [0.1, 0.2],
                [0.3, 0.4]
            ])

            self.db = np.array([
                [0.05, 0.10]
            ])


    layer = DummyLayer()

    optimizer = SGD(lr=0.1)

    print("Before Update")
    print("Weights:\n", layer.weight)
    print("Biases:\n", layer.biases)

    optimizer.update(layer)

    print("\nAfter Update")
    print("Weights:\n", layer.weight)
    print("Biases:\n", layer.biases)
