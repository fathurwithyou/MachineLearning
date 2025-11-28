from abc import ABC, abstractmethod
from ..base import BaseEstimator, TransformerMixin


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):

    @abstractmethod
    def fit(self, X, y):
        return self

    @abstractmethod
    def transform(self, X):
        return self
