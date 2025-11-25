import numpy as np
from ..base import BaseEstimator, TransformerMixin

class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2'):
        super().__init__()
        self.norm = norm

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X, dtype=float)

        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        norms[norms == 0] = 1.0
        return X / norms
