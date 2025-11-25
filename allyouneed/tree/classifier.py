import numpy as np
from collections import Counter
from .decision_tree import DecisionTree
from ..base import ClassifierMixin

class DecisionTreeClassifier(DecisionTree, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        super().fit(X, y)
        return self

    def _calculate_leaf_value(self, y):
        """Returns the most common class in the node"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.array(X)
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")
        return np.array([self._traverse_tree_proba(x, self.root) for x in X])

    def _traverse_tree_proba(self, x, node):
        if node.value is not None:
            proba = np.zeros(len(self.classes_))
            for label in node.samples:
                class_idx = np.where(self.classes_ == label)[0][0]
                proba[class_idx] += 1
            proba /= len(node.samples)
            return proba

        if x[node.feature] < node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)