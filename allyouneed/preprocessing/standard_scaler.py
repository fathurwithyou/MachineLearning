import numpy as np
from ..base import BaseEstimator, TransformerMixin

class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = np.array(X)

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = None

        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X, dtype=float)

        if self.with_mean:
            X = X - self.mean_

        if self.with_std:
            X = X / self.scale_

        return X
