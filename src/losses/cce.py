import numpy as np

sm_op = np.array([0.75, 0.1, 0.15])
tg_op = np.array([1, 0, 0])

cce = -np.sum(tg_op * np.log(sm_op))
print(cce)