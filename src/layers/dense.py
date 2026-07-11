""" Implementation of fully connected dense layer """
import numpy as np

class Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weight = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.rand(1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.biases
        return self.output

if __name__ == "__main__":
    np.random.seed(11)
    # Sample batch containing 4 input vectors, each with 5 features.
    X = np.array([
        [1.0, 2.0, 3.0, 2.5, 1.5],
        [2.0, 5.0, 1.0, 2.0, 4.0],
        [3.0, 6.0, 2.0, 1.0, 5.0],
        [4.0, 7.0, 3.0, 2.0, 6.0]
    ])

    layer = Dense(5, 6)
    output = layer.forward(X)

    print("Weights:\n",layer.weight)
    print("\nBiases:\n",layer.biases)
    print("\nOutput:\n",output)
