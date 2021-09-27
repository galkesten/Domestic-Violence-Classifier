import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold


def get_best_params_mlp(x_train, Y):
    mlp_gs = MLPClassifier(max_iter=10000, random_state=42)
    parameter_space = {
        'activation': ['identity', 'logistic', 'relu', 'softmax', 'tanh'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'learning_rate': ['constant','adaptive', 'invscaling'],
        'hidden_layer_sizes': [(50, 100, 50), (50, 50, 50), (100,), (100, 1)],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=cv)
    grid.fit(x_train, Y)
    print("Best parameters for MLP are %s with a score of %f"
          % (grid.best_params_, grid.best_score_))


def get_best_params_svm(x_train, Y):
    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range, random_state=[20, 42])
    param_grid = {'C': np.array([0.01, 0.1, 1, 10, 100, 1000]),
                'kernel': np.array(['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': ['auto', 'scale', 1, 0.5, 0.1, 0.01, 0.001, 0.0001]}
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x_train, Y)

    print("The best parameters for SVM are %s with a score of %f"
          % (grid.best_params_, grid.best_score_))


def get_best_params_rf(x_train, Y):
    param_grid = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [10, 20, 50, 70, 100, 200]
                  }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=cv)
    grid.fit(x_train, Y)

    print("The best parameters for RF are %s with a score of %f"
          % (grid.best_params_, grid.best_score_))


