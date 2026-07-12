import numpy as np

class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input

        for l in self.layers:
            output = l.forward(output)

        return output

    def backward(self, dval):
        grad = dval

        for l in reversed(self.layers):
            l.backward(grad)
            grad = l.dinputs

    def predict(self, input):
        return self.forward(input)

if __name__ == "__main__":

    from src.layers.dense import Dense
    from src.activations.relu import ReLU

    np.random.seed(11)

    X = np.random.randn(4, 5)

    model = Model()

    model.add(Dense(5, 6))
    model.add(ReLU())
    model.add(Dense(6, 3))

    output = model.forward(X)

    print("\nModel Output:\n", output)
