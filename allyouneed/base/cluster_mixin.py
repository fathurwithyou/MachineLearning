import numpy as np

class ClusterMixin:
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)

        unique_labels = np.unique(labels)
        inertia = 0

        for label in unique_labels:
            cluster_points = X[labels == label]
            if hasattr(self, 'cluster_centers_'):
                center = self.cluster_centers_[label]
            else:
                center = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - center) ** 2)

        return -inertia
