import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier
from joblib import Parallel, delayed

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []
        
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_ = int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                self.max_features_ = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, float):
            self.max_features_ = int(self.max_features * self.n_features_)
        else:
            self.max_features_ = self.max_features

        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y) for _ in range(self.n_estimators)
        )
        return self
    
    def _build_tree(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_X = X[idxs]
        bootstrap_y = y[idxs]
        
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        tree.max_features = self.max_features_
        return tree.fit(bootstrap_X, bootstrap_y)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return majority vote
        return np.array([
            np.bincount(predictions[:, i]).argmax()
            for i in range(len(X))
        ])
    
    def feature_importances_(self):
        importances = np.zeros(self.n_features_)
        for tree in self.trees:
            importances += self._feature_importance_tree(tree)
        return importances / len(self.trees)
    
    def _feature_importance_tree(self, tree):
        importances = np.zeros(self.n_features_)
        def _traverse(node, importance):
            if node.feature is not None: 
                importances[node.feature] += importance
                _traverse(node.left, importance / 2)
                _traverse(node.right, importance / 2)
        _traverse(tree.root, 1.0)
        return importances