from abc import ABC, abstractmethod

class TransformerMixin(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(self, X):
        ...

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)