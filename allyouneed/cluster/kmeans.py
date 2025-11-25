import numpy as np
from ..base import BaseEstimator, ClusterMixin

class KMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.array(X)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices].copy()

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)

            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            center_shift = np.sum(np.sqrt(np.sum((new_centers - self.cluster_centers_) ** 2, axis=1)))

            self.cluster_centers_ = new_centers

            if center_shift < self.tol:
                break

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)

        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.array(X)
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.cluster_centers_) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X, labels):
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.cluster_centers_[k]) ** 2)
        return inertia
