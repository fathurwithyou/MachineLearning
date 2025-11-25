import numpy as np
from ..tree import DecisionTreeRegressor
from ..base import BaseEstimator, RegressorMixin

class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_ = X.shape[1]

        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_ = int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                self.max_features_ = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, float):
            self.max_features_ = int(self.max_features * self.n_features_)
        else:
            self.max_features_ = self.max_features

        self.trees = []
        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def _build_tree(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_X = X[idxs]
        bootstrap_y = y[idxs]

        feature_idxs = np.random.choice(self.n_features_, self.max_features_, replace=False)
        X_subset = bootstrap_X[:, feature_idxs]

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        tree.feature_idxs = feature_idxs
        tree.fit(X_subset, bootstrap_y)
        return tree

    def predict(self, X):
        self._check_is_fitted()
        X = np.array(X)

        predictions = []
        for tree in self.trees:
            X_subset = X[:, tree.feature_idxs]
            predictions.append(tree.predict(X_subset))

        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)
