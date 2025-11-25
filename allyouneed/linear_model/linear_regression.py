from .base import LinearModel
from ..base import RegressorMixin
import numpy as np

class LinearRegression(LinearModel, RegressorMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        self.is_fitted = True
        return self
    
    def predict(self, X):
        self._check_is_fitted()
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        return X_b.dot(np.r_[self.intercept_, self.coef_])
    