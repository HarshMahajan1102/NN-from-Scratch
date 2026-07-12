import numpy as np

class Accuracy:

    def calculate(self, y_p, y_t):
        pred = np.argmax(y_p, axis=1)

        if len(y_t.shape) == 2:
            y_t = np.argmax(y_t, axis=1)

        return np.mean(pred == y_t)
