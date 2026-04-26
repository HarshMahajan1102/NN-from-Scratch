# Trying to implement a batch of 4 inputs of 5 features and feeding it to 6 neurons
import numpy as np

X = [[1, 2, 3, 2.5, 1.5],
     [2, 5, 1, 2, 4],
     [3, 6, 2, 1, 5],
     [4, 7, 3, 2, 6]]

np.random.seed(11)

class layer:
	def __init__(self, n_inputs, n_neurons):
		self.weight = np.random.randn(n_inputs, n_neurons)
		self.biases = np.random.rand(1, n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weight) + self.biases
	def printProp(self):
		print("Weights: ")
		print(self.weight)
		print("Biases: ", self.biases)
		print("Output: ")
		print(self.output)

l = layer(5, 6)
l.forward(X)
l.printProp()