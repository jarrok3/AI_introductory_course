from random_forest_algo import (
    load_and_prepare_data,
    train_random_forest,
    evaluate_model,
    plot_roc_curve,
    optimize_hyperparameters
)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data('lab-4-dataset.csv')

    model = train_random_forest(X_train, y_train)

    y_prob = evaluate_model(model, X_test, y_test)

    plot_roc_curve(y_test, y_prob)

    best_model = optimize_hyperparameters(X_train, y_train)
    y_best_prob = evaluate_model(best_model, X_test, y_test)
    plot_roc_curve(y_test, y_best_prob, title='ROC Curve (po optymalizacji)')
