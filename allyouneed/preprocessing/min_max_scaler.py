import numpy as np
from ..base import BaseEstimator, TransformerMixin

class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = np.array(X)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.data_range_ = self.max_ - self.min_
        self.data_range_[self.data_range_ == 0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X, dtype=float)

        X_scaled = (X - self.min_) / self.data_range_
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return X_scaled
