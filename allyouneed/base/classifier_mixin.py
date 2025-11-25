import numpy as np

class ClassifierMixin:
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
