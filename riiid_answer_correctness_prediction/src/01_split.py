import os
import gc
import cudf
import numpy as np
import pandas as pd

from cuml.preprocessing.model_selection import train_test_split

# Ref: https://www.kaggle.com/marisakamozz/cv-strategy-in-the-kaggle-environment


def fast_merge(left, right, key):
    return cudf.concat(
        [
            left.reset_index(drop=True),
            right.reindex(left[key].values).reset_index(drop=True),
        ],
        axis=1,
    )


def calc_max_timestamps(train):
    a = 2.2
    b = 2.3

    max_timestamp_u = train[["user_id", "timestamp"]].groupby(["user_id"]).max()
    max_timestamp_u.columns = ["max_timestamp"]
    max_timestamp_u["interval"] = (
        max_timestamp_u.max_timestamp.max() - max_timestamp_u.max_timestamp
    )
    # max_timestamp_u['random'] = np.random.rand(len(max_timestamp_u))
    max_timestamp_u["random"] = np.random.beta(a, b, len(max_timestamp_u))
    max_timestamp_u["random_timestamp"] = (
        max_timestamp_u.interval * max_timestamp_u.random
    )
    max_timestamp_u["random_timestamp"] = max_timestamp_u.random_timestamp.astype(int)
    max_timestamp_u.drop(["interval", "random"], axis=1, inplace=True)
    return max_timestamp_u


def main():
    src_dir = "../data/raw/"
    dst_dir = "../data/01_split/"

    train = cudf.read_csv(
        os.path.join(src_dir, "train.csv"),
        dtype={
            "row_id": "int64",
            "timestamp": "int64",
            "user_id": "int32",
            "content_id": "int16",
            "content_type_id": "int8",
            "task_container_id": "int16",
            "user_answer": "int8",
            "answered_correctly": "int8",
            "prior_question_elapsed_time": "float32",
            "prior_question_had_explanation": "boolean",
        },
    )

    max_timestamp_u = calc_max_timestamps(train)

    train = fast_merge(train, max_timestamp_u, "user_id")
    train["virtual_timestamp"] = train.timestamp + train.random_timestamp
    train.set_index(["virtual_timestamp", "row_id"], inplace=True)
    train.sort_index(inplace=True)
    train.reset_index(inplace=True)
    train.drop(columns=["max_timestamp", "random_timestamp"], inplace=True)

    print(max_timestamp_u.describe())
    print(train.head())

    # Dump evaluation data.
    eval_size = 1_000_000
    eval_data = train[-eval_size:]
    train_data = train[:-eval_size]

    eval_data.to_parquet(os.path.join(dst_dir, "evaluation.parquet"))

    # Dump train and validation data.
    val_size = 2_500_000
    for n_fold in range(5):
        valid_data = train_data[-val_size:]
        train_data = train_data[:-val_size]
        valid_data.to_parquet(os.path.join(dst_dir, f"fold_{n_fold}_valid.parquet"))
        train_data.to_parquet(os.path.join(dst_dir, f"fold_{n_fold}_train.parquet"))


if __name__ == "__main__":
    main()
