from typing import List, Dict

import numpy as np
import pandas as pd

from .base import BaseTrainer

from pytorch_tabnet.tab_model import TabNetClassifier


class TabNetClassificationTrainer(BaseTrainer):
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(
        self,
        params: Dict,
        train_params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        categorical_feature: List[str] = None,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):
        self.feature_names = X_valid.columns.tolist()

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().ravel()
        X_valid = X_valid.to_numpy()
        y_valid = y_valid.to_numpy().ravel()

        self.model = TabNetClassifier(**params)
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=["train", "valid"],
            **train_params,
        )

    def predict(self, data):
        pred = self.model.predict_proba(data.to_numpy())[:, 1]
        return pred

    def get_importance(self):
        """Return feature importance.

        Returns
        -------
        dict :
            Dictionary of feature name, feature importance.
        """
        importance = np.zeros(len(self.feature_names))
        feature_name = self.feature_names

        return dict(zip(feature_name, importance))

    def get_model(self):
        return self.model
