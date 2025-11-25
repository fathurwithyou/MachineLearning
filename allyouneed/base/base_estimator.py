from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        return self

    @abstractmethod
    def predict(self, X):
        ...

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("This estimator has not been fitted yet.")
        return True
