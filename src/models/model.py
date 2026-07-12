import numpy as np

class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else []

    def forward(self, X):
        output = X

        for l in self.layers:
            output = l.forward(output)

        return output

    def backward(self, dval):
        grad = dval

        for l in reversed(self.layers):
            l.backward(grad)
            
            if hasattr(l, "di"):
                grad = l.di 

    def predict(self, X, return_classes=False):

        output = self.forward(X)
        if return_classes:
            return np.argmax(output, axis=1)
        return output
    
    def _require_compiled(self, action):
        if not hasattr(self, "loss") or not hasattr(self, "optimizer"):
            raise RuntimeError(f"Model must be compiled before calling {action}(). Call model.compile(loss=..., optimizer=...) first.")
     
    def _train_step(self, X, y):
        output = self.forward(X)
        loss = self.loss.forward(output, y)
        self.loss.backward(self.loss.output, y)
        self.backward(self.loss.di)

        for l in self.layers:
            if hasattr(l, "weight"):
                self.optimizer.update(l)

        m_scores = {}
        for m in self.metrics:
            key = m.__class__.__name__.lower()
            m_scores[key] = m.calculate(output, y)

        return loss, m_scores

    def fit(self, X, y, es=100, batch_size=None, verbose=True):
    
        self._require_compiled("fit")
        n_samples = X.shape[0]

        history = {"loss": []}
        for m in self.metrics:
            history[m.__class__.__name__.lower()] = []

        for e in range(1, epochs + 1):
            if batch_size is None:
                e_loss, e_metrics = self._train_step(X, y)
            else:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                b_losses = []
                b_m_score = {m.__class__.__name__.lower(): [] for m in self.metrics}

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    loss, m_scores = self._train_step(X_batch, y_batch)
                    b_losses.append(loss)
                    for key, value in m_scores.items():
                        b_m_score[key].append(value)

                e_loss = np.mean(b_losses)
                e_metrics = {key: np.mean(values) for key, values in b_m_score.items()}

            history["loss"].append(e_loss)
            for key, value in e_metrics.items():
                history[key].append(value)

            if verbose:
                log = f"Epoch {e}/{epochs} | Loss: {e_loss:.5f}"
                for key, value in e_metrics.items():
                    log += f" | {key}: {value:.4f}"
                print(log)

        return history 

    def evaluate(self, X, y):

        self._require_compiled("evaluate")
        output = self.forward(X)
        loss = self.loss.forward(output, y)

        results = {"loss": loss}
        log = f"Loss: {loss:.5f}"

        for m in self.metrics:
            score = m.calculate(output, y)
            key = m.__class__.__name__.lower()
            results[key] = score
            log += f" | {key}: {score:.4f}"

        print(log)
        return results

if __name__ == "__main__":
    from src.layers.dense import Dense
    from src.activations.relu import ReLU
    from src.losses.softmax_cce import SoftmaxCCE
    from src.optimizers.adam import Adam
    from src.metrics.accuracy import Accuracy

    np.random.seed(11)

    X = np.random.randn(8, 5)
    y = np.eye(3)[np.random.randint(0, 3, size=8)]

    model = Model()
    model.add(Dense(5, 6))
    model.add(ReLU())
    model.add(Dense(6, 3))

    model.compile(loss=SoftmaxCCE(), optimizer=Adam(), metrics=[Accuracy()])

    history = model.fit(X, y, epochs=10, batch_size=4, verbose=True)

    print("\nEvaluation:")
    model.evaluate(X, y)
