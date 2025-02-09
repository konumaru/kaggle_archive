import pickle
from typing import Callable

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
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


class Amex_tabnet(Metric):
    def __init__(self):
        self._name = "amex_tabnet"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        amex = amex_metric_mod(y_true, y_pred[:, 1])
        return max(amex, 0.0)


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
        "D_67",
        "D_68",
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    # X.loc[:, cat_features] = X[cat_features].astype("category")

    oof_predictions = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(X, y)):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {X.shape[1]} features...")

        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        if fold <= -1:
            with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "rb") as f:
                model = pickle.load(f)
        else:
            model = TabNetClassifier(
                n_d=32,
                n_a=32,
                n_steps=3,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                momentum=0.02,
                clip_value=None,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-3, weight_decay=1e-3),
                scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                scheduler_params={
                    "T_0": 5,
                    "eta_min": 1e-4,
                    "T_mult": 1,
                    "last_epoch": -1,
                },
                mask_type="entmax",
                seed=seed,
            )

            model.fit(
                np.array(x_train),
                np.array(y_train.values.ravel()),
                eval_set=[(np.array(x_val), np.array(y_val.values.ravel()))],
                max_epochs=60,
                patience=50,
                batch_size=512,
                eval_metric=["auc", "accuracy", Amex_tabnet],
            )
            with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "wb") as f:
                pickle.dump(model, f)

        pred_val = model.predict_proba(x_val.to_numpy())[:, 1]
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

    cols_to_drop = get_low_variance_columns(
        train.select_dtypes(include=["float32"]), threshold=0.005
    )
    train.drop(cols_to_drop, axis=1, inplace=True)
    print(f"Drop {len(cols_to_drop)} feature by variance.")

    return train


def main():
    train = pd.read_parquet("./output/train_fe.parquet").head(10000)
    train = feature_selection(train)
    features = [col for col in train.columns if col not in ["customer_ID", "target"]]

    train.fillna(0, inplace=True)

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
