from scikit_algo import (
    load_and_prepare_data,
    train_random_forest,
    evaluate_model,
    plot_roc_curve,
    optimize_hyperparameters,
    create_forest_n_trees
)

import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from random_forest_algo import RandomForest

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('invalid arg no.')
        sys.exit()
    
    X_train, X_test, y_train, y_test = load_and_prepare_data('lab-4-dataset.csv')
       
    if sys.argv[1] == 'scikit': 
        model = train_random_forest(X_train, y_train)
        
        # Create forest with n trees
        n = 20
        fixed_model, fixed_pred, fixed_prob = create_forest_n_trees(
            X_train, y_train, X_test, n_estimators=n
        )
        evaluate_model(fixed_model, X_test, y_test)
        plot_roc_curve(y_test, fixed_prob, title=f'ROC Curve ({n} drzew)')

        # Before optimization
        y_prob = evaluate_model(model, X_test, y_test)
        plot_roc_curve(y_test, y_prob)

        # After optimization
        best_model = optimize_hyperparameters(X_train, y_train)
        y_best_prob = evaluate_model(best_model, X_test, y_test)
        plot_roc_curve(y_test, y_best_prob, title='ROC Curve (po optymalizacji)')

    else:
    #     for n in [10, 50, 100, 200]:
        for n in [20]:
            rf_model = RandomForest(n_estimators=n)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            y_scores = rf_model.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=rf_model.positive_class)
            rec = recall_score(y_test, y_pred, pos_label=rf_model.positive_class)
            fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=rf_model.positive_class)
            auc_val = auc(fpr, tpr)
            
            print(f"n_estimators = {n}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, AUC = {auc_val:.4f}")
            
            plot_roc_curve(y_test,y_scores)