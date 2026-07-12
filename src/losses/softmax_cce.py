from src.activations.softmax import Softmax
from src.losses.cce import CCE

import numpy as np

class SoftmaxCCE:

    def __init__(self):
        self.activation = Softmax()
        self.loss = CCE()

    def forward(self, inputs, y_t):
        self.output = self.activation.forward(inputs)
        return self.loss.forward(self.output, y_t)

    def backward(self, dval, y_t):

        samp = len(dval)

        if len(y_t.shape) == 2:
            y_t=np.argmax(y_t, axis=1)

        self.di = dval.copy()
        self.di[range(samp), y_t] -= 1
        self.di = self.di/samp
