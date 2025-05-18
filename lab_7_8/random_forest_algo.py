import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath : str) -> list:
    """load from .csv file and returns data split into training and test set

    Args:
        filepath (_type_): .csv file path

    Returns:
        _type_: x_train,x_test,y_train,y_test
    """
    df = pd.read_csv(filepath)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X) # standardize data for data integrity guarantee
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train) # fit to the training set
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] #probability of prediction correctness

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Wyniki klasyfikacji ===")
    print(f"Dokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision): {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"Pole pod krzywą ROC (AUC): {auc:.4f}")

    return y_prob

def plot_roc_curve(y_test, y_prob, title :str ='ROC Curve') -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def optimize_hyperparameters(X_train, y_train):
    param_grid = {'n_estimators': [10, 50, 100, 200, 300, 400, 500]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)
    print(f"\nNajlepsze parametry: {grid_search.best_params_}")
    return grid_search.best_estimator_

def create_forest_n_trees(X_train, y_train, X_test, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob