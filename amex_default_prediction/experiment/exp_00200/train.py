import gc
import os
import pickle
from typing import Callable, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def amex_metric_mod(y_true, y_pred):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return "amex_metric", amex_metric_mod(y_true, y_pred), True


def train_lgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    # X.loc[:, cat_features] = X[cat_features].astype("category")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        # "metric": "None",
        "boosting": "dart",
        "seed": seed,
        "num_leaves": 100,
        "learning_rate": 0.01,
        "feature_fraction": 0.50,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "device_type": "cpu",
    }
    oof_predictions = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(X, y)):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {X.shape[1]} features...")

        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
        lgb_valid = lgb.Dataset(x_val, y_val, free_raw_data=False)

        del x_train, y_train
        gc.collect()

        if fold <= 0:
            with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "rb") as f:
                model = pickle.load(f)
        else:
            model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_valid],
                feval=lgb_amex_metric,
                callbacks=[
                    lgb.log_evaluation(200),
                    # lgb.early_stopping(500),
                ],
            )
            with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "wb") as f:
                pickle.dump(model, f)

        pred_val = model.predict(x_val)
        oof_predictions[val_ind] = pred_val

    pd.Series(oof_predictions, name="oof").to_csv(
        f"./output/oof_seed={seed}.csv", index=False
    )

    return oof_predictions


def train_rsa(
    X: pd.DataFrame,
    y: pd.Series,
    num_seed: int,
    train_fn: Callable,
    seed: int = 42,
):

    rsa_oof_predictions = np.zeros(len(X))
    for i in range(num_seed):
        sub_seed = seed + i
        print(f"\n\n=== Training with seed={sub_seed} ===")

        # oof = train_fn(X, y, sub_seed)
        oof = pd.read_csv(f"./output/oof_seed={sub_seed}.csv")["oof"].to_numpy()
        rsa_oof_predictions += oof / num_seed

    return rsa_oof_predictions


def feature_selection(train):
    print("\n\n=== Feature Selection ===")

    def get_low_variance_columns(data: pd.DataFrame, threshold: int = 0.0):
        cols = data.columns[(data.var(axis=0) <= threshold)].tolist()
        return cols

    cols_to_drop = get_low_variance_columns(
        train.select_dtypes(include=["float32"]), threshold=0.005
    )
    train.drop(cols_to_drop, axis=1, inplace=True)
    print(f"Drop {len(cols_to_drop)} feature by variance.")

    return train


def main():
    train = pd.read_parquet("./output/train_fe.parquet")
    train = feature_selection(train)
    features = [col for col in train.columns if col not in ["customer_ID", "target"]]

    with open("./output/train_features.pkl", "wb") as f:
        pickle.dump(train.columns.tolist(), f)

    # oof_predictions = train_lgb(X=train[features], y=train["target"], seed=46)
    oof_predictions = train_rsa(
        X=train[features], y=train["target"], num_seed=5, train_fn=train_lgb, seed=42
    )

    # NOTE: Compute out of folds metric.
    oof_predictions = pd.Series(oof_predictions, name="oof")
    oof_predictions.to_csv("./output/oof.csv", index=False)

    score = amex_metric_mod(train["target"].to_numpy(), oof_predictions.to_numpy())

    print(f"Our out of folds CV score is {score}")

    with open(f"score={score:.6f}.txt", "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
