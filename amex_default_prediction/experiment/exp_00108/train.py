import pickle
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


def xgb_amex(y_pred, y_true):
    return "amex", amex_metric_mod(y_true.get_label(), y_pred)


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


def train_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "booster": "gbtree",
        "max_depth": 7,
        "subsample": 0.5,
        "colsample_bytree": 0.7,
        "gamma": 1.5,
        "min_child_weight": 100,
        "lambda": 50,  # 70,
        "eta": 0.05,
        "scale_pos_weight": 2.8,
        "disable_default_eval_metric": 1,
        "random_state": seed,
    }
    oof_predictions = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(X, y)):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {X.shape[1]} features...")

        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dvalid = xgb.DMatrix(data=x_val, label=y_val)

        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        bst = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=1600,
            evals=watchlist,
            # early_stopping_rounds=500,
            custom_metric=xgb_amex,
            maximize=True,
            verbose_eval=200,
        )
        bst.save_model(f"./output/xgb_seed={seed}_fold={fold}.json")

        pred_val = bst.predict(dvalid, iteration_range=(0, bst.best_ntree_limit))
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

        oof = train_fn(X, y, sub_seed)
        # oof = pd.read_csv(f"./output/oof_seed={sub_seed}.csv")["oof"].to_numpy()
        rsa_oof_predictions += oof / num_seed

    return rsa_oof_predictions


def feature_selection(train):
    print("\n\n=== Feature Selection ===")

    def get_low_variance_columns(data: pd.DataFrame, threshold: int = 0.0):
        cols = data.columns[(data.var(axis=0) <= threshold)].tolist()
        return cols

    # cols_to_drop = get_low_variance_columns(
    #     train.select_dtypes(include=["float32"]), threshold=0.005
    # )

    cols_to_drop = get_low_variance_columns(
        train.select_dtypes(include=["float32"]), threshold=0.0
    )
    train.drop(cols_to_drop, axis=1, inplace=True)
    print(f"Drop {len(cols_to_drop)} feature by variance.")

    return train


def main():
    # train = pd.read_parquet("./output/train_fe.parquet")
    train = pd.read_parquet("./output/train_fe_pca1800.parquet")
    train = feature_selection(train)
    features = [col for col in train.columns if col not in ["customer_ID", "target"]]

    with open("./output/train_features.pkl", "wb") as f:
        pickle.dump(train.columns.tolist(), f)

    # oof_predictions = train_xgb(X=train[features], y=train["target"], seed=46)
    oof_predictions = train_rsa(
        X=train[features], y=train["target"], num_seed=5, train_fn=train_xgb, seed=42
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
