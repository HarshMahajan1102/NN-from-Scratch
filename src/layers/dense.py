"""Implementation of fully connected dense layer"""
import numpy as np

class Dense:

    def __init__(self, n_inputs, n_neurons, init="random"):
        if init == "random":
            self.weight = np.random.randn(n_inputs, n_neurons)
        elif init == "xavier":
            self.weight = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif init == "he":
            self.weight = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError("Initialization must be 'random', 'xavier', or 'he'.")
    
        self.biases = np.random.rand(1, n_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weight) + self.biases
        return self.output
    
    def backward(self, dval):
        self.dw=np.dot(self.inputs.T, dval)
        self.db=np.sum(dval, axis=0, keepdims=True)
        self.di=np.dot(dval, self.weight.T)

if __name__ == "__main__":
    np.random.seed(11)
    # Sample batch containing 4 input vectors, each with 5 features.
    X = np.array(
        [
            [1.0, 2.0, 3.0, 2.5, 1.5],
            [2.0, 5.0, 1.0, 2.0, 4.0],
            [3.0, 6.0, 2.0, 1.0, 5.0],
            [4.0, 7.0, 3.0, 2.0, 6.0],
        ]
    )

    layer = Dense(5, 6)
    output = layer.forward(X)

    print("Weights:\n", layer.weight)
    print("\nBiases:\n", layer.biases)
    print("\nOutput:\n", output)

    dval = np.random.randn(*output.shape)
    layer.backward(dval)

    print("\nWeight Gradients:\n", layer.dw)
    print("\nBias Gradients:\n", layer.db)
    print("\nInput Gradients:\n", layer.di)
