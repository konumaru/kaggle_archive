import os
import pathlib
import pickle
from cProfile import label
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
NUM_FOLD = 5
NUM_SEED = 5

data_dir = pathlib.Path("../data/")

asset_details = pd.read_csv("../data/raw/asset_details.csv")


def dump_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def wmean(x, w):
    return np.sum(x * w) / np.sum(w)


def wcov(x, y, w):
    return np.sum(w * (x - wmean(x, w)) * (y - wmean(y, w))) / np.sum(w)


def wcorr(x, y, w):
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))


def eval_wcorr(preds, y_true, weight):
    return "eval_wcorr", wcorr(preds, y_true, weight), True


def eval_wcorr_xgb(pred: np.ndarray, data: xgb.DMatrix):
    labels = data.get_label()
    weight = data.get_weight()

    metric = wcorr(pred, labels, weight)
    return "wcorr", metric


def fit_lgb(X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed):
    model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        metric="None",
        learning_rate=1e-7,
        n_estimators=500,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.4,
        reg_alpha=1,
        reg_lambda=1,
        importance_type="gain",
    )

    model.set_params(seed=seed)
    model.fit(
        X_train,
        Y_train,
        sample_weight=train_weight,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        eval_sample_weight=[train_weight, valid_weight],
        eval_names=["train", "valid"],
        eval_metric=eval_wcorr,  # "rmse",
        callbacks=[
            lgb.log_evaluation(period=20),
            lgb.early_stopping(stopping_rounds=20),
        ],
    )
    return model


def fit_xgb(X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed):
    train = xgb.DMatrix(X_train, label=Y_train, weight=train_weight)
    valid = xgb.DMatrix(X_valid, label=Y_valid, weight=valid_weight)

    params = dict(
        objective="reg:squarederror",
        verbosity=0,
        disable_default_eval_metric=True,
        eta=3e-5,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.5,
        tree_method="gpu_hist",
        predictor="cpu_predictor",
        feature_selector="greedy",
        random_state=seed,
    )
    model = xgb.train(
        params,
        train,
        num_boost_round=1000,
        evals=[(valid, "valid")],
        maximize=True,
        feval=eval_wcorr_xgb,
        early_stopping_rounds=20,
        verbose_eval=50,
    )
    return model


def fit_mlp(X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed):
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 256),
        ),
    )
    model.fit(X_train, Y_train)
    return model


def train(
    model_name: str,
    n_fold: int,
):
    X_train = pd.read_pickle(data_dir / f"feature/fold{n_fold:02}_train.pkl")
    Y_train = pd.read_pickle(data_dir / f"split/fold{n_fold:02}_train_target.pkl")
    train_weight = pd.read_pickle(data_dir / f"split/fold{n_fold:02}_train_weight.pkl")
    X_valid = pd.read_pickle(data_dir / f"feature/fold{n_fold:02}_valid.pkl")
    Y_valid = pd.read_pickle(data_dir / f"split/fold{n_fold:02}_valid_target.pkl")
    valid_weight = pd.read_pickle(data_dir / f"split/fold{n_fold:02}_valid_weight.pkl")

    X_train = X_train[~Y_train.isnull()]
    train_weight = train_weight[~Y_train.isnull()]
    Y_train = Y_train[~Y_train.isnull()]

    X_valid = X_valid[~Y_valid.isnull()]
    valid_weight = valid_weight[~Y_valid.isnull()]
    Y_valid = Y_valid[~Y_valid.isnull()]

    model_dir = data_dir / f"model/{model_name}/"
    os.makedirs(model_dir, exist_ok=True)
    for i in range(NUM_SEED):
        seed = SEED + i
        if model_name == "lgb":
            model = fit_lgb(
                X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed
            )
        elif model_name == "xgb":
            model = fit_xgb(
                X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed
            )
        elif model_name == "mlp":
            model = fit_mlp(
                X_train, Y_train, X_valid, Y_valid, train_weight, valid_weight, seed
            )

        dump_pickle(
            model,
            model_dir / f"seed{seed:03}_fold{n_fold:02}.pkl",
        )


def main():
    for n_fold in range(NUM_FOLD):
        # train("lgb", n_fold)
        # train("xgb", n_fold)
        # train("mlp", n_fold)


if __name__ == "__main__":
    main()
