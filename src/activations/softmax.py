import numpy as np
np.random.seed(11)

inputs = [[0.3, 2.9, 4.0, -0.5],
          [1.0, 0.5, -1.0, 2.0],
          [-1.5, 2.3, 0.0, 3.1],
          [3.2, -0.1, 0.0, 1.5],
          [2.0, 1.0, -2.0, 0.5]]

class layer:
	def __init__(self, n_inputs, n_neurons):
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.random.rand(n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		return self.output

l1 = layer(4,6)
output = l1.forward(inputs)

output = output - np.max(output, axis=1, keepdims=True)

# print(output)
# print()

exp_val = np.exp(output)
# print(exp_val)

# norm_val = exp_val/np.sum(exp_val)
norm_val = exp_val/np.sum(exp_val, axis=1, keepdims=True)

print(norm_val)
# print(np.sum(norm_val, axis=1, keepdims=True))