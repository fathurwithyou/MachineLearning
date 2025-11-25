import numpy as np

class RegressorMixin:
    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
