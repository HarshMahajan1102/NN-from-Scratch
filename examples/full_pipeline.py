import numpy as np
np.random.seed(11)

inputs = [[0.3, 2.9, 4.0, -0.5],
          [1.0, 0.5, -1.0, 2.0],
          [-1.5, 2.3, 0.0, 3.1],
          [3.2, -0.1, 0.0, 1.5],
          [2.0, 1.0, -2.0, 0.5]]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.rand(n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

def ReLU(in_vals):
    return np.maximum(0, in_vals)

def softMax(in_vals):
    in_vals -= np.max(in_vals, axis=1, keepdims=True)
    exp_vals = np.exp(in_vals)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def cce(pred_op, tar_op):
    return -np.sum(tar_op * np.log(pred_op), axis=1, keepdims=True)

l1 = Layer(4, 3)
l1_op = l1.forward(inputs)
l1_op_act = ReLU(l1_op)

l2 = Layer(3, 4)
l2_op = l2.forward(l1_op_act)
l2_op_act = softMax(l2_op)

print("Output:\n", l2_op_act)

target = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]])

loss = cce(l2_op_act, target)
print("\nLoss:\n", loss)
