import numpy as np
from .base import LinearModel
from ..base import ClassifierMixin

class LogisticRegression(LinearModel, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        for _ in range(self.max_iter):
            linear_pred = np.dot(X, self.coef_) + self.intercept_
            y_pred = self._sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.array(X)
        linear_pred = np.dot(X, self.coef_) + self.intercept_
        return self._sigmoid(linear_pred)

    def predict(self, X):
        self._check_is_fitted()
        return (self.predict_proba(X) >= 0.5).astype(int)
