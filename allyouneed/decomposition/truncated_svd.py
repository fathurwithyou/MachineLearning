import numpy as np
from ..base import BaseEstimator, TransformerMixin

class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_iter=5, random_state=None):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.array(X)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(S ** 2 / (X.shape[0] - 1))

        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X)

        return np.dot(X, self.components_.T)

    def inverse_transform(self, X_transformed):
        self._check_is_fitted()
        X_transformed = np.array(X_transformed)

        return np.dot(X_transformed, self.components_)
