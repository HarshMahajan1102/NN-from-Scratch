import numpy as np

class Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dval):
        self.di = dval * (1 - self.output ** 2)

if __name__ == "__main__":
    X = np.array([
        [-2.0, 5.0, 3.0, -1.0, 0.0],
        [1.0, -4.0, 6.0, 2.0, -3.0]
    ])

    tanh = Tanh()
    output = tanh.forward(X)

    dval = np.ones_like(output)
    tanh.backward(dval)

    print("Forward Output:\n", output)
    print("\nBackward Output:\n", tanh.di)
