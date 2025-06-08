# NUMERIC LIBRARIES IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# DATA IMPORTS
from ucimlrepo import fetch_ucirepo 

class BayesClassificator:
    """
    Naive Bayes classificator.
    @classMethods:
        fit() -> gets probabilities from training data\n
        predict() -> predicts classes
        
    @staticMethods:
        evaluate() -> evaluates the accuracy based on the predicted and actual data\n
        get_data() -> downloads data from Spambase source
    """
    def __init__(self) -> None:
        self.class_prob : dict = {}
        self.class_stats : dict = {}
    
    def fit(self, X:pd.DataFrame, y:pd.Series) -> None:
        """Using the provided test set, get the probabilities of P(C) and P(x_i|C)
 
        Args:
            X (pd.DataFrame): test set features
            y (pd.DataFrame): test set classes
        """
        
        assert isinstance(X,pd.DataFrame,), "features must be pd.DataFrame type"
        assert isinstance(y,pd.Series,), "class (target) must be pd.Series type"
        
        class_counts = y.value_counts(normalize=True)
        self.class_prob = class_counts.to_dict()

        for class_label in class_counts.index:
            X_class = X[y == class_label]

            self.class_stats[class_label] = {
                'mean': X_class.mean(),
                'std': X_class.std(ddof=0) + 1e-9
            }

    def _gaussian_density(self, x: float, mean: float, std: float) -> float:
        """
        Returns the probability density of x under Gaussian distribution with given mean and std.
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Predict the class label for each sample in X_test.
        """
        predictions = []

        for _, row in X_test.iterrows():
            class_probs  = {}

            for class_label in self.class_prob:
                prob = self.class_prob[class_label]
                for feature in X_test.columns:
                    mean = self.class_stats[class_label]['mean'][feature]
                    std = self.class_stats[class_label]['std'][feature]
                    prob *= self._gaussian_density(row[feature], mean, std)
                class_probs[class_label] = prob

            predicted = max(class_probs , key=class_probs.get)
            predictions.append(predicted)

        return pd.Series(predictions, index=X_test.index)
        
    @staticmethod
    def evaluate(y_test:pd.Series, y_pred:pd.Series) -> float:
        return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0)
        }
        
    @staticmethod
    def get_data(test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get source data from Spambase and divide it into training and test sets.
        """
        spambase = fetch_ucirepo(id=94)
        X = spambase.data.features
        y = spambase.data.targets.iloc[:, 0]

        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    @staticmethod
    def evaluate_random_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, test_size: float = 0.2) -> pd.DataFrame:
        """
        Evaluate model using repeated random train-test splits.
        Returns DataFrame with metrics from each run.
        """
        results = []

        for i in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=i
            )
            model = BayesClassificator()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = BayesClassificator.evaluate(y_test, y_pred)
            metrics["split"] = i
            results.append(metrics)

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_kfold(X: pd.DataFrame, y: pd.Series, k: int = 25) -> pd.DataFrame:
        """
        Evaluate model using K-Fold cross-validation.
        Returns DataFrame with metrics from each fold.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = BayesClassificator()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = BayesClassificator.evaluate(y_test, y_pred)
            metrics["fold"] = i
            results.append(metrics)

        return pd.DataFrame(results)