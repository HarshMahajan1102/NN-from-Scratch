import numpy as np

class ReLU:

    def forward(self, inputs):
        self.inputs=inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dval):
        self.di=dval.copy()
        self.di[self.inputs<=0]=0

if __name__ == "__main__":
    X = np.array([
        [-2, 5, 3, -1, 0],
        [1, -4, 6, 2, -3]
    ])

    relu = ReLU()

    output = relu.forward(X)

    dval = np.ones_like(output)

    relu.backward(dval)

    print("Forward Output:\n", output)
    print("\nBackward Output:\n", relu.di)
