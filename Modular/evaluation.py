import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_roc_curve



# Grid Search to find the best model and the best parameters
def grid_search(parameters: list(dict), classifier, scoring: list, k_value: int, X_train: np.ndarray, y_train: np.ndarray):
    grid_search_res = GridSearchCV(estimator = classifier,
                                param_grid = parameters,
                                scoring = scoring,
                                cv = k_value,
                                n_jobs = -1)
    grid_search_res.fit(X_train, y_train)
    best_accuracy = grid_search_res.best_score_
    best_parameters = grid_search_res.best_params_
    print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
    print("Best Parameters:", best_parameters)

#K fold Cross Validation
def k_fold_cv(classifier, k_value: int, X_train: np.ndarray, y_train: np.ndarray):
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = k_value)
    print("10 Fold Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


def model_evaluation(classifier, y_test: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score= (2* precision * recall) / (precision + recall)

    plot_roc_curve(classifier, X_test, y_test) 
    print(f"Precision : {precision*100}\n Recall: {recall*100}\n F1 Score: {f1_score*100}\n Accuracy: {accuracy*100}\n")

