import numpy as np

class Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dval):
        self.di = dval * self.output * (1 - self.output)

if __name__ == "__main__":

    X = np.array([
        [-2.0, 5.0, 3.0, -1.0, 0.0],
        [1.0, -4.0, 6.0, 2.0, -3.0]
    ])

    sigmoid = Sigmoid()
    output = sigmoid.forward(X)

    dval = np.ones_like(output)
    sigmoid.backward(dval)

    print("Forward Output:\n", output)
    print("\nBackward Output:\n", sigmoid.di)
