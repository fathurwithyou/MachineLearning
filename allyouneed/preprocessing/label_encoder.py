import numpy as np
from ..base import BaseEstimator, TransformerMixin

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, y, X=None):
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self

    def transform(self, y):
        self._check_is_fitted()
        y = np.array(y)
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def inverse_transform(self, y):
        self._check_is_fitted()
        y = np.array(y)
        return self.classes_[y]
