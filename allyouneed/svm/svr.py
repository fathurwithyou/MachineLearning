import numpy as np
from ..base import BaseEstimator, RegressorMixin

class SVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', epsilon=0.1, max_iter=1000, tol=1e-3):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X1.shape[1] * X1.var())
            elif self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** 3
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.alpha_star = np.zeros(n_samples)
        self.b = 0

        K = self._kernel_function(X, X)

        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            alpha_star_prev = np.copy(self.alpha_star)

            for i in range(n_samples):
                f_i = np.sum((self.alpha - self.alpha_star) * K[:, i]) + self.b
                E_i = f_i - y[i]

                if (E_i > self.epsilon and self.alpha[i] < self.C) or \
                   (E_i < -self.epsilon and self.alpha_star[i] < self.C):

                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    f_j = np.sum((self.alpha - self.alpha_star) * K[:, j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_star_i_old = self.alpha_star[i]

                    if E_i > self.epsilon:
                        delta_alpha = min(self.C - self.alpha[i], (E_i - self.epsilon) / (K[i, i] + K[j, j]))
                        self.alpha[i] += delta_alpha
                    elif E_i < -self.epsilon:
                        delta_alpha = min(self.C - self.alpha_star[i], (-E_i - self.epsilon) / (K[i, i] + K[j, j]))
                        self.alpha_star[i] += delta_alpha

                    self.b = y[i] - np.sum((self.alpha - self.alpha_star) * K[:, i])

            if np.linalg.norm(self.alpha - alpha_prev) < self.tol and \
               np.linalg.norm(self.alpha_star - alpha_star_prev) < self.tol:
                break

        sv = (self.alpha > 1e-5) | (self.alpha_star > 1e-5)
        self.support_vectors_ = X[sv]
        self.support_alpha_ = (self.alpha - self.alpha_star)[sv]

        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.array(X)
        K = self._kernel_function(X, self.support_vectors_)
        return np.sum(self.support_alpha_ * K.T, axis=0) + self.b
