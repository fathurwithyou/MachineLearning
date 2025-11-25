from ..base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import Counter

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
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
            nn_labels = self.y_train[nn_indices]
            most_common = Counter(nn_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)