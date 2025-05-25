import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neural import Network

if __name__ == "__main__":
    df = pd.read_csv("lab_9_10/wsi5-25L_dataset.csv")
    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    # Normalize data for features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    
    # Get output classes
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y] # classes vector of quality values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    training_data   = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_train, y_train)]
    validation_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_val, y_val)]
    test_data       = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X_test, y_test)]

    num_features = X_train.shape[1]
    num_classes = y_train.shape[1]
    net = Network([num_features, 128, 64, 32, 16, num_classes]) # 3 layers
    
    max_steps = 10000
    batch_size = 32
    initial_learning_rate = 0.01
    decay_rate = 0.95
    decay_steps = 1000
    patience = 20  # Number of steps to wait for improvement
    best_loss = float('inf')
    steps_without_improvement = 0
    
    for step in range(max_steps):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
        for mini_batch in mini_batches:
            net.gradient_descent(mini_batch, max_steps=1, learning_rate = initial_learning_rate * (decay_rate ** (step // decay_steps)))
        
        # Calculate validation loss
        validation_loss = 0
        for x, y in validation_data:
            output = net.feedforward(x)
            validation_loss += net.cost.entropy_cost(output, y)
        validation_loss /= len(validation_data)
        
        print(f"Step {step+1}, Validation Loss: {validation_loss:.4f}")
        
        # Check for improvement
        if validation_loss < best_loss:
            best_loss = validation_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        # Stop if no improvement for 'patience' steps
        if steps_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    correct = net.evaluate(test_data)
    total   = len(test_data)
    accuracy = correct / total * 100
    print(f"Accuracy on test set: {correct}/{total} correct ({accuracy:.2f}%)")


