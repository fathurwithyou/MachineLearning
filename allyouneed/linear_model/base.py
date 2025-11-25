from ..base import BaseEstimator
from abc import abstractmethod

class LinearModel(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass