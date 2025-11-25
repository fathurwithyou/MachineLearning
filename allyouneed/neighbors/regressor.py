from ..base import BaseEstimator, RegressorMixin
import numpy as np

class KNeighborsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nn_indices = np.argsort(distances)[:self.n_neighbors]
            nn_values = self.y_train[nn_indices]
            predictions.append(np.mean(nn_values))
        return np.array(predictions)