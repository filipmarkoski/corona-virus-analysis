import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from definitions import RNG_SEED


class AveragingModels(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights=None):
        self.models_ = []
        self.models = models
        if weights is not None:
            self.weights = weights
        else:
            n = len(models)
            self.weights = [1 / n] * n

    # Define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        weight = list()
        predictions = np.array([model.predict(X) for model in self.models_])

        # for every data point, single model prediction times weight, then add them together
        for data in range(predictions.shape[1]):
            single = [predictions[model, data] * weight for model, weight in
                      zip(range(predictions.shape[0]), self.weights)]
            weight.append(np.sum(single))
        return weight


class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in self.kf.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.base_models)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean


class BestModels:
    """
    ('kernel_ridge', 0.109341336)
    ('svr', 0.10940373022194627)
    ('ridge', 0.1108020002843447)
    ('elastic_net', 0.11169509602064272)
    ('lasso', 0.1117297704490916)
    ('huber_regressor', 0.1122407889111273)
    ('xgb_regressor', 0.1577407481325562)
    ('lgbm_regressor', 0.16356273159774978)
    """

    def __init__(self):
        # {'alpha': 0.33, 'coef0': 1, 'degree': 3, 'kernel': 'polynomial'} 0.109341336
        self.kernel_ridge = KernelRidge(alpha=0.33, kernel='polynomial', degree=3, coef0=1)
        # {'alpha': 0.0008, 'max_iter': 10000} 0.1117297704490916
        self.lasso = Lasso(alpha=0.0008, max_iter=10000)
        # {'alpha': 56.32} 0.1108020002843447
        self.ridge = Ridge(alpha=56.32)
        # {'C': 11, 'epsilon': 0.01, 'gamma': 0.0004, 'kernel': 'rbf'} 0.10940373022194627
        self.svr = SVR(C=11, epsilon=0.01, gamma=0.0004, kernel='rbf')
        # {'alpha': 0.01, 'l1_ratio': 0.08, 'max_iter': 10000} 0.11169509602064272
        self.elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.08, max_iter=10000)
        # {'gamma': 0.6, 'learning_rate': 0.75, 'max_depth': 3, 'min_child_weight': 1.78, 'n_estimators': 40, 'n_jobs': -1,
        # 'objective': 'reg:squarederror', 'random_state': 1, 'reg_alpha': 0.04, 'reg_lambda': 0.75,
        # 'silent': True} 0.1577407481325562
        self.xgb_regressor = xgb.XGBRegressor(n_estimators=40, n_jobs=-1, objective='reg:squarederror',
                                              gamma=0.6, learning_rate=0.75, max_depth=3,
                                              random_state=RNG_SEED, reg_alpha=0.04, reg_lambda=0.75, silent=True)
        # {'bagging_freq': 5, 'bagging_seed': 9, 'feature_fraction': 0.225, 'feature_fraction_seed': 9,
        # 'learning_rate': 0.75, 'max_bin': 100, 'min_child_weight': 1.78, 'min_data_in_leaf': 6,
        # 'min_sum_hessian_in_leaf': 11, 'n_estimators': 40, 'n_jobs': -1,
        # 'num_leaves': 5, 'objective': 'regression', 'random_state': 1, 'silent': True} 0.16356273159774978
        self.lgbm_regressor = lgb.LGBMRegressor(bagging_freq=5, bagging_seed=9, feature_fraction=0.225,
                                                feature_fraction_seed=9,
                                                learning_rate=0.75, max_bin=100, min_child_weight=1.78,
                                                min_data_in_leaf=6,
                                                min_sum_hessian_in_leaf=11, n_estimators=40, n_jobs=-1, num_leaves=5,
                                                objective='regression', random_state=1, silent=True)
        # {'alpha': 0.04, 'epsilon': 1.0} 0.1122407889111273
        self.huber_regressor = HuberRegressor(alpha=0.04, epsilon=1)

        # xgb_regressor, lgbm_regressor

        self.models = [self.lasso, self.ridge, self.svr, self.elastic_net, self.huber_regressor]
