import numpy as np

class Dropout:

    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, inputs, training=True):

        self.inputs = inputs

        if training:
            self.mask = (np.random.rand(*inputs.shape) > self.rate) / (1 - self.rate)
            self.output = inputs * self.mask
        else:
            self.mask = np.ones_like(inputs)
            self.output = inputs

        return self.output

    def backward(self, dval):
        self.di = dval * self.mask

if __name__ == "__main__":
    np.random.seed(11)
    X = np.random.randn(4, 5)

    dropout = Dropout(rate=0.5)

    output_train = dropout.forward(X, training=True)
    output_eval = dropout.forward(X, training=False)

    dval = np.ones_like(output_train)
    dropout.backward(dval)

    print("Training Output:\n", output_train)
    print("\nEval Output:\n", output_eval)
    print("\nBackward Output:\n", dropout.di)
