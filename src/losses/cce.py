import numpy as np

class CCE:
    def forward(self, y_p, y_t):
        y_p=np.clip(y_p, 1e-7, 1-1e-7)
        losses = -np.sum(y_t * np.log(y_p), axis=1)
        loss = np.mean(losses)
        self.output=loss 
        return self.output
