import math
import random
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature       
        self.threshold = threshold    
        self.value = value            
        self.children = {}           
        self.left = None             
        self.right = None            

class DecisionTree:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.root = None
        self.target = None  
    
    def entropy(self, y_series : pd.Series):
        counts = y_series.value_counts(normalize=True)
        entropy = 0.0
        for p in counts:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def build_tree(self, data, features):
        if data.shape[0] == 0:
            return None
        y = data[self.target]
        # STOP
        # When leaf reached
        if y.nunique() == 1:
            return Node(value=y.iloc[0])
        # Or no more features
        if not features:
            majority_class = y.value_counts().idxmax()
            return Node(value=majority_class)
        
        # Prepare children creation
        parent_entropy = self.entropy(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_splits = None
        features_to_consider = features
        
        # Provide part of features for tree
        if self.max_features is not None:
            m = self.max_features if isinstance(self.max_features, int) else len(features)
            if m < len(features):
                features_to_consider = random.sample(features, m) # randomize
            else:
                features_to_consider = features
                
        # Best division choice
        for feat in features_to_consider:
            unique_vals = np.sort(data[feat].unique()) # values for feature
            if unique_vals.size <= 1:
                continue  
            for i in range(unique_vals.size - 1):
                thr = (unique_vals[i] + unique_vals[i+1]) / 2.0
                left_data = data[data[feat] < thr]
                right_data = data[data[feat] >= thr]
                if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                    continue  
                left_entropy = self.entropy(left_data[self.target])
                right_entropy = self.entropy(right_data[self.target])
                
                w_entropy = (left_data.shape[0] / data.shape[0]) * left_entropy \
                            + (right_data.shape[0] / data.shape[0]) * right_entropy
                info_gain = parent_entropy - w_entropy
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feat
                    best_threshold = thr
                    best_splits = (left_data, right_data)
                    
        
        if best_feature is None or best_gain <= 0:
            # if no way of better info gains, create leaf
            majority_class = y.value_counts().idxmax()
            return Node(value=majority_class)
        node = Node(feature=best_feature, threshold=best_threshold)
        remaining_features = [f for f in features if f != best_feature]  
        if best_threshold is not None:
            left_data, right_data = best_splits
            node.left = self.build_tree(left_data, remaining_features)
            node.right = self.build_tree(right_data, remaining_features)
        else:
            node.children = {}
            for val, subset in best_splits.items():
                node.children[val] = self.build_tree(subset, remaining_features)
        return node
    
    def fit(self, X, y, target_name="HeartDisease"):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X)
        data[target_name] = y.values if isinstance(y, pd.Series) else y
        self.target = target_name
        features = [col for col in data.columns if col != target_name]
        # Build tree
        self.root = self.build_tree(data, features)
    
    def predict_one(self, x):
        node = self.root
        while node is not None:
            # If in leaf, return its value
            if node.feature is None or node.value is not None:
                return node.value
            
            # Get record val
            val = x[node.feature] if isinstance(x, (pd.Series, dict)) else x[node.feature]

            # Traversew
            if val < node.threshold:
                node = node.left
            else:
                node = node.right
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda row: self.predict_one(row), axis=1).values
        else:
            X_df = pd.DataFrame(X)
            return X_df.apply(lambda row: self.predict_one(row), axis=1).values

