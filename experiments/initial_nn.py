# Trying to implement a layer from 5 inputs to 3 neurons

import numpy as np 
inputs = [0.5, 0.2, 0.1, 0.4, 0.8]  
weights = [[0.5, 0.75, -0.6, 1.85, -1.5],  
           [0.5, -0.91, 0.26, -0.5, 0.1],
           [-0.25, -0.2, 0.17, 0.87, 0.5]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases

print(outputs)