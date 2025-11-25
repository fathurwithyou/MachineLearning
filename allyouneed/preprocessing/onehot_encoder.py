import numpy as np
from ..base import BaseEstimator, TransformerMixin

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sparse=False):
        super().__init__()
        self.sparse = sparse

    def fit(self, X, y=None):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for i in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, i]))

        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded = []
        for i in range(X.shape[1]):
            n_categories = len(self.categories_[i])
            feature_encoded = np.zeros((X.shape[0], n_categories))

            for j, value in enumerate(X[:, i]):
                idx = np.where(self.categories_[i] == value)[0]
                if len(idx) > 0:
                    feature_encoded[j, idx[0]] = 1

            encoded.append(feature_encoded)

        return np.hstack(encoded)

    def inverse_transform(self, X):
        self._check_is_fitted()
        X = np.array(X)

        original = []
        col_idx = 0

        for i in range(len(self.categories_)):
            n_categories = len(self.categories_[i])
            feature_onehot = X[:, col_idx:col_idx + n_categories]
            feature_original = self.categories_[i][np.argmax(feature_onehot, axis=1)]
            original.append(feature_original)
            col_idx += n_categories

        return np.column_stack(original)
