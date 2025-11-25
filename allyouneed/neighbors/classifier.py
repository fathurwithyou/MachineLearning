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
        self.classes_ = np.unique(y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        probabilities = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nn_indices = np.argsort(distances)[:self.n_neighbors]
            nn_labels = self.y_train[nn_indices]

            proba = np.zeros(len(self.classes_))
            for label in nn_labels:
                class_idx = np.where(self.classes_ == label)[0][0]
                proba[class_idx] += 1
            proba /= self.n_neighbors
            probabilities.append(proba)
        return np.array(probabilities)

    def predict(self, X):
        self._check_is_fitted()
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]