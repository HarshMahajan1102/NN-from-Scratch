import numpy as np

class Adam:

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, ep=1e-7):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.ep = ep
        self.it = 0

    def update(self, layer):

        if not hasattr(layer, "w_momentum"):
            layer.w_momentum = np.zeros_like(layer.weight)
            layer.b_momentum = np.zeros_like(layer.biases)

            layer.w_cache = np.zeros_like(layer.weight)
            layer.b_cache = np.zeros_like(layer.biases)

        layer.w_momentum = (self.b1*layer.w_momentum+(1-self.b1)*layer.dw)
        layer.b_momentum = (self.b1*layer.b_momentum+(1-self.b1)*layer.db)

        w_momentum_corrected=(layer.w_momentum/(1-self.b1**(self.it+1)))
        b_momentum_corrected=(layer.b_momentum/(1-self.b1**(self.it+1)))

        layer.w_cache=(self.b2*layer.w_cache+(1-self.b2)*(layer.dw**2))
        layer.b_cache=(self.b2*layer.b_cache+(1-self.b2)*(layer.db**2))

        w_cache_corrected=(layer.w_cache/(1-self.b2**(self.it+1)))
        b_cache_corrected=(layer.b_cache/(1-self.b2**(self.it+1)))

        layer.weight -= (self.lr*w_momentum_corrected/(np.sqrt(w_cache_corrected)+self.ep))
        layer.biases -= (self.lr*b_momentum_corrected/(np.sqrt(b_cache_corrected)+self.ep))

        self.it += 1

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

    optimizer = Adam()

    print("Before Update")
    print("Weights:\n", layer.weight)
    print("Biases:\n", layer.biases)

    optimizer.update(layer)

    print("\nAfter Update")
    print("Weights:\n", layer.weight)
    print("Biases:\n", layer.biases)
