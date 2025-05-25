import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neural import Network

if __name__ == "__main__":
    df = pd.read_csv("wsi5-25L_dataset.csv")
    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    # Normalize data for features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    
    # Get output classes
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y] # classes vector of quality values

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    # X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    training_data   = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_train, y_train)]
    # validation_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_val, y_val)]
    test_data       = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_test, y_test)]

    num_features = X_train.shape[1]
    num_classes = y_train.shape[1]
    net = Network([num_features, 16, num_classes]) # 3 layers
    
    net.gradient_descent(training_data, max_steps=10000, learning_rate=0.1)
    correct = net.evaluate(test_data)
    total   = len(test_data)
    accuracy = correct / total * 100
    print(f"Accuracy on test set: {correct}/{total} correct ({accuracy:.2f}%)")


