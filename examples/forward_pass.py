from src.layers.dense import Dense
from src.activations.relu import ReLU
from src.activations.softmax import Softmax
from src.losses.cce import CCE

import numpy as np

np.random.seed(11)

X = np.array([
    [1.0, 2.0, 3.0, 2.5, 1.5],
    [2.0, 5.0, 1.0, 2.0, 4.0],
    [3.0, 6.0, 2.0, 1.0, 5.0],
    [4.0, 7.0, 3.0, 2.0, 6.0]
])

y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

dense1 = Dense(5, 6)
relu = ReLU()

dense2 = Dense(6, 3)
softmax = Softmax()

loss_fn = CCE()

d1_output = dense1.forward(X)
relu_output = relu.forward(d1_output)

d2_output = dense2.forward(relu_output)
out = softmax.forward(d2_output)

loss = loss_fn.forward(out, y)

print("Dense Layer 1 Output:")
print(d1_output)

print("\nReLU Output:")
print(relu_output)

print("\nDense Layer 2 Output:")
print(d2_output)

print("\nSoftmax Probabilities:")
print(out)

print("\nRow Sums:")
print(np.sum(out, axis=1))

print("\nCategorical Cross Entropy Loss:")
print(loss)
