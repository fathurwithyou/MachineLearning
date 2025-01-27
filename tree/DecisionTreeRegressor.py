import numpy as np
from DecisionTree import DecisionTree

class DecisionTreeRegressor(DecisionTree):
    def _calculate_leaf_value(self, y):
        """Returns the mean value of the node"""
        return np.mean(y)

    def _information_gain(self, y, X_column, threshold):
        parent_var = self._variance(y)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -float('inf')

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        var_l, var_r = self._variance(y[left_idxs]), self._variance(y[right_idxs])
        
        # Calculate weighted variance of children
        weighted_var = (n_l/n) * var_l + (n_r/n) * var_r

        # Return variance reduction
        return parent_var - weighted_var

    def _variance(self, y):
        """Calculate variance of the target values"""
        return np.var(y) if len(y) > 1 else 0