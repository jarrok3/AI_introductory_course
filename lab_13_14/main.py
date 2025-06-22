from naive_bayes import BayesClassificator
import pandas as pd

if __name__=="__main__":
    x_train, x_test, y_train, y_test = BayesClassificator.get_data()
    
    classificator = BayesClassificator()
    classificator.fit(x_train,y_train)
    predictions = classificator.predict(x_test)
    
    metrics = BayesClassificator.evaluate(y_test, predictions)
    
    print(f"Accuracy: {metrics["accuracy"]*100:.2f}%\nPrecision: {metrics["precision"]*100:.2f}%\nRecall: {metrics["recall"]*100:.2f}%")
    
    X = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])
    
    print("\nRandom Splits Evaluation:")
    rand_results = BayesClassificator.evaluate_random_splits(X, y)
    print(rand_results.describe().T)
    
    print("\nK-Fold Evaluation:")
    kfold_results = BayesClassificator.evaluate_kfold(X, y)
    print(kfold_results.describe().T)