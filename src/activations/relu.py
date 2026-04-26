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
outputs = l1.forward(inputs)
print(outputs)
print()

def ReLU(output):
    return np.maximum(0, output)

outputs = ReLU(outputs)
print(outputs)