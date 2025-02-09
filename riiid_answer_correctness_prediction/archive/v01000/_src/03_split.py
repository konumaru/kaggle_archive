import os
import gc
import cudf
import dask_cudf
import dask.dataframe as dd

import numpy as np
import pandas as pd

from sklearn import model_selection

from utils.common import timer
from utils.common import dump_pickle
from utils.fold import StratifiedGroupKFold

"""
Todo:
- https://www.kaggle.com/its7171/cv-strategy
"""


def main():
    train = dask_cudf.read_csv("../data/train_dataset/train_dataset_data_*.csv")
    train = train.fillna(-1)
    train = train.compute().to_pandas()

    print(train.shape)
    print(train.columns)
    print(train.head())

    drop_cols = [
        "user_id",
        "answered_correctly",
        "lag_answered_correctly",
        "lag_content_id",
        "lag_content_type_id",
        "lag_task_container_id",
        "reci_past_total_answered_count",
    ]
    train_cols = train.columns.tolist()
    features = [c for c in train_cols if c not in drop_cols]
    target = "answered_correctly"

    dump_pickle(features, "../data/split/feature_names.pkl")

    X = train[features]
    y = train[target]
    weight = train["reci_past_total_answered_count"].clip(1e-5, 1.0)
    groups = train["user_id"].to_numpy()

    print(X.shape)
    print(X.head())
    del train
    gc.collect()

    dump_dir = "../data/split/"
    # Split and dump by fold.
    cv = model_selection.GroupKFold(n_splits=5)
    # cv = model_selection.StratifiedKFold(n_splits=5)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        X_train.to_pickle(os.path.join(dump_dir, f"{n_fold}_fold_X_train.pkl"))
        y_train.to_pickle(os.path.join(dump_dir, f"{n_fold}_fold_y_train.pkl"))
        X_valid.to_pickle(os.path.join(dump_dir, f"{n_fold}_fold_X_valid.pkl"))
        y_valid.to_pickle(os.path.join(dump_dir, f"{n_fold}_fold_y_valid.pkl"))

        weight_train = weight.iloc[train_idx]
        weight_valid = weight.iloc[valid_idx]

        weight_train.to_pickle(
            os.path.join(dump_dir, f"{n_fold}_fold_weight_train.pkl")
        )
        weight_valid.to_pickle(
            os.path.join(dump_dir, f"{n_fold}_fold_weight_valid.pkl")
        )

        print(f"Dump files {n_fold} fold.")
        print("train taget mean:", y.iloc[train_idx].mean())
        print("valid target mean:", y.iloc[valid_idx].mean())
        print("")


if __name__ == "__main__":
    with timer("CV Split"):
        main()
