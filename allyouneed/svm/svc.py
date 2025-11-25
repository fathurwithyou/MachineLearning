import numpy as np
from ..base import BaseEstimator, ClassifierMixin

class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', max_iter=1000, tol=1e-3):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
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

        y = np.where(y <= 0, -1, 1)

        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0

        K = self._kernel_function(X, X)

        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):

                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    E_j = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break

        sv = self.alpha > 1e-5
        self.support_vectors_ = X[sv]
        self.support_labels_ = y[sv]
        self.support_alpha_ = self.alpha[sv]

        self.is_fitted = True
        return self

    def decision_function(self, X):
        self._check_is_fitted()
        X = np.array(X)
        K = self._kernel_function(X, self.support_vectors_)
        return np.sum(self.support_alpha_ * self.support_labels_ * K.T, axis=0) + self.b

    def predict_proba(self, X):
        self._check_is_fitted()
        decision = self.decision_function(X)
        prob_positive = 1 / (1 + np.exp(-decision))
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    def predict(self, X):
        self._check_is_fitted()
        decision = self.decision_function(X)
        return np.where(decision <= 0, 0, 1)
