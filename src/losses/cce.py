import numpy as np

class CCE:

    def forward(self, y_p, y_t):
        self.output = -np.sum(y_t * np.log(y_p))
        return self.output
