import numpy as np
from ..base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.array(X)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is None:
            self.n_components = X.shape[1]

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)

        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X)

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        self._check_is_fitted()
        X_transformed = np.array(X_transformed)

        return np.dot(X_transformed, self.components_) + self.mean_
