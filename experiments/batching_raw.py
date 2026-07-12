# Trying to implement a batch of 4 inputs of 5 features and feeding it to 6 neurons
import numpy as np

inputs = [[1, 2, 3, 2.5, 1.5], [2, 5, 1, 2, 4], [3, 6, 2, 1, 5], [4, 7, 3, 2, 6]]

weights = [
    [0.2, 0.8, -0.5, 1, 0.5],
    [0.5, -0.91, 0.26, -0.5, 1.5],
    [-0.26, -0.27, 0.17, 0.87, 0.1],
    [0.1, 0.14, 0.5, -0.91, -0.5],
    [0.2, 0.8, -0.5, 1, 0.5],
    [0.5, -0.91, 0.26, -0.5, 1.5],
]

biases = [2, 3, 0.5, 1, 2, 3]

output = np.dot(inputs, np.array(weights).T) + biases
print(output)

# Further implementing another layer with 3 neurons

weights2 = [
    [0.1, -0.14, 0.5, -0.91, -0.5, 0.26],
    [-0.5, 0.91, -0.26, 0.5, -1.5, 0.87],
    [0.26, 0.27, -0.17, -0.87, -0.1, 0.5],
]

biases2 = [-1, 2, -0.5]
output2 = np.dot(output, np.array(weights2).T) + biases2
print(output2)
