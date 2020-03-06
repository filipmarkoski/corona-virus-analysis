import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from definitions import RNG_SEED, SERIALIZED_MODELS_DIR


def rmse(targets, predictions):
    return np.sqrt(mean_squared_error(targets, predictions))


def rmsle(targets, predictions):
    return np.sqrt(mean_squared_log_error(targets, predictions))


def rmsle_cv(model, train, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=RNG_SEED).get_n_splits(train.X)
    rmse = np.sqrt(-cross_val_score(model, train.X, train.y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()


def formatted_score_line(dataset_name, metric_name, score):
    return f'{dataset_name:15} {metric_name:10} = {score:20.7}'


def compute_score(model, dataset, dataset_name, include_cv=True):
    results = [
        formatted_score_line(dataset_name, 'RMSE', rmse(model.predict(dataset.X), dataset.y)),
    ]

    if hasattr(model, 'score'):
        results.extend([
            formatted_score_line(dataset_name, 'Score', model.score(dataset.X, dataset.y)),
        ])

    if hasattr(model, 'oob_score_'):
        results.extend([
            formatted_score_line(dataset_name, 'oob_score', model.oob_score_)
        ])

    if include_cv:
        results.extend([
            formatted_score_line(dataset_name, 'RMSLE_CV', rmsle_cv(model, dataset))
        ])

    return results


def print_score(model, train, valid=None, include_valid=False, include_train=True, include_cv=False):
    results = []

    if include_valid:
        valid_results = compute_score(model, valid, 'Validation Set', include_cv)
        results.extend(valid_results)
    else:
        include_cv = True

    if include_train:
        train_results = compute_score(model, train, 'Training Set', include_cv)
        results.extend(train_results)

    for result in results:
        print(f'\t{result}')


def calculate_classification_metrics(test_targets, predictions) -> dict:
    """Calculation of accuracy score, F1 micro and F1 macro"""
    results = {
        'accuracy': accuracy_score(test_targets, predictions),
        'f1-micro': f1_score(test_targets, predictions, average="micro"),  # good for unbalanced classes
        'f1-macro': f1_score(test_targets, predictions, average="macro"),
        'precision': precision_score(test_targets, predictions, average="weighted"),
        'recall': recall_score(test_targets, predictions, average="weighted"),
    }
    # 'accuracy', 'f1-micro', 'f1-macro', 'precision', 'recall'
    print(f"\tAccuracy score: {results['accuracy']:.3f}")
    print(f"\tF1-micro: {results['f1-micro']:.3f}")
    print(f"\tF1-macro: {results['f1-macro']:.3f}")
    print(f"\tPrecision score: {results['precision']:.3f}")
    print(f"\tRecall score: {results['recall']:.3f}")

    return results


def feature_importances(model, feature_list: list, importance_limit: float = 0.05, display_top_n: int = 20,
                        display_graph=True):
    importances = list(model.feature_importances_)  # List of tuples with variable and importance

    # Sort the feature importances by most important first
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)  # Print out the feature and importances

    features_to_keep = []
    for i, pair in zip(range(display_top_n), feature_importances):
        feature, importance = pair
        if importance < importance_limit:
            break
        features_to_keep.append(feature)
        print(f'{i + 1:3}. Feature: {feature:20} Importance: {importance}')

    print(features_to_keep)

    if display_graph:
        features = list()
        importances = list()

        for feature, importance in feature_importances[:display_top_n]:
            features.append(feature)
            importances.append(importance)

        # convert to pd.DataFrame
        sns.set(style='darkgrid')
        plt.plot(features, importances)
        plt.xticks(rotation=90)
        plt.show()
    return features_to_keep


class Grid:
    def __init__(self, model):
        self.model = model

    def search(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error",
                                   verbose=True)
        grid_search.fit(X, y)

        print(f'Best model: {grid_search.best_params_} with score={np.sqrt(-grid_search.best_score_):.6}')

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        results: pd.DataFrame = pd.DataFrame(grid_search.cv_results_)

        print('Cross Validation results for all grid search candidates')
        print(results[['params', 'mean_test_score', 'std_test_score']])

        # Save the Grid Search results to a CSV file
        model = grid_search.best_estimator_
        results.to_csv(f'{SERIALIZED_MODELS_DIR}/{model.__class__.__name__}_{timestamp()}.csv')
        return grid_search


def make_grid(model, train, param_grid):
    grid = Grid(model=model)
    grid_search = grid.search(X=train.X, y=train.y, param_grid=param_grid)
    return grid_search.best_estimator_


def timestamp():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
