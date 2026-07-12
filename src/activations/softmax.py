import numpy as np


class Softmax:
    def forward(self, inputs):
        shifted_ip = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_val = np.exp(shifted_ip)
        norm_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = norm_val
        return self.output
