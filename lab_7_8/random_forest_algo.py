import pandas as pd
import math
import numpy as np
from decision_tree import DecisionTree
from collections import Counter
class RandomForest:
    def __init__(self, n_estimators=100, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.feature_count = None
        self.classes = None
        self.positive_class = None
    
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = pd.Series(y)
        
        n_feats = X.shape[1]

        m = self.max_features
        if m is None:
            m = int(math.sqrt(n_feats))
            m = max(m, 1)

        self.feature_count = n_feats
        self.classes = np.unique(y)

        if len(self.classes) == 2:
            self.positive_class = sorted(self.classes)[1]
        else:
            self.positive_class = None

        self.trees = []
        n_samples = len(y)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            tree = DecisionTree(max_features=m)
            tree.fit(X_sample, y_sample, target_name="HeartDisease")
            self.trees.append(tree)
    
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        all_preds = [tree.predict(X) for tree in self.trees]
        all_preds = np.array(all_preds)

        final_preds = []
        for j in range(all_preds.shape[1]):
            votes = all_preds[:, j]
            final_preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_preds)
    
    def predict_proba(self, X):
        if len(self.classes) != 2:
            raise ValueError("predict_proba dziala binarnie")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        all_preds = [tree.predict(X) for tree in self.trees]
        all_preds = np.array(all_preds)
        return np.mean(all_preds == self.positive_class, axis=0)
