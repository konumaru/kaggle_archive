import numpy as np
import pandas as pd
from typing import List, Dict


class BaseTrainer(object):
    def __init__(self):
        self.model = None

    def fit(
        self,
        params: Dict,
        train_params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        weight_train: pd.DataFrame = None,
        weight_valid: pd.DataFrame = None,
    ):
        NotImplementedError

    def predict(self, data):
        NotImplementedError

    def get_importance(self):
        NotImplementedError

    def get_model(self):
        NotImplementedError
