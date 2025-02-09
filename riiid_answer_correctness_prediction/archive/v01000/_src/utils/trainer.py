from typing import List

import xgboost as xgb
import lightgbm as lgb


class BaseTrainer(object):
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, X_valid, y_valid):
        NotImplementedError

    def predict(self, data):
        NotImplementedError

    def get_importance(self):
        NotImplementedError

    def get_model(self):
        NotImplementedError


class XGBTrainer(BaseTrainer):
    def __init__(self):
        self.model = None

    def fit(
        self,
        params,
        train_params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        weight_train=None,
        weight_valid=None,
    ):
        train_dataset = xgb.DMatrix(X_train, label=y_train, weight=weight_train)
        valid_dataset = xgb.DMatrix(X_valid, label=y_valid, weight=weight_valid)

        self.model = xgb.train(
            params,
            train_dataset,
            evals=[(train_dataset, "train"), (valid_dataset, "valid")],
            **train_params
        )

    def predict(self, data):
        pred = self.model.predict(
            xgb.DMatrix(data), ntree_limit=self.model.best_ntree_limit
        )
        return pred

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        return self.model.get_score(importance_type="gain")

    def get_model(self):
        return self.model


class LGBTrainer(BaseTrainer):
    def __init__(self):
        self.model = None

    def fit(
        self,
        params,
        train_params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_feature: List[str] = None,
        weight_train=None,
        weight_valid=None,
    ):
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=weight_train,
            categorical_feature=categorical_feature,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=weight_valid,
        )

        self.model = lgb.train(
            params, train_data, valid_sets=[train_data, valid_data], **train_params
        )

    def predict(self, data):
        pred = self.model.predict(data, num_iteration=self.model.best_iteration)
        return pred

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = self.model.feature_importance(
            importance_type="gain", iteration=self.model.best_iteration
        )
        feature_name = self.model.feature_name()

        return dict(zip(feature_name, importance))

    def get_model(self):
        return self.model
