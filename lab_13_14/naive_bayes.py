# NUMERIC LIBRARIES IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# DATA IMPORTS
from ucimlrepo import fetch_ucirepo 

class BayesClassificator:
    """
    Naive Bayes classificator.
    @classMethods:
        fit() -> gets probabilities from training data
        predict() -> predicts classes
        
    @staticMethods:
        evaluate() -> evaluates the accuracy based on the predicted and actual data
        get_data() -> downloads data from Spambase source
    """
    def __init__(self) -> None:
        pass
    
    @classmethod
    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        """Using the provided test set, get the probabilities of P(C) and P(x_i|C)
 
        Args:
            X (pd.DataFrame): test set features
            y (pd.DataFrame): test set class
        """
        
        assert isinstance(X,pd.DataFrame,), "features must be pd.DataFrame type"
        assert isinstance(y,pd.DataFrame,), "class (target) must be pd.DataFrame type"
        
    @classmethod
    def predict(self, X_test:pd.DataFrame):
        pass
        
    @staticmethod
    def evaluate(y_test:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        pass
        
    @staticmethod
    def get_data(test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get source data from Spambase and divide it into training and test sets.
        """
        spambase = fetch_ucirepo(id=94)
        X = spambase.data.features
        y = spambase.data.targets

        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    