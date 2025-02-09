import os
import random

import cudf
import dask_cudf
import numpy as np
import pandas as pd

from utils.common import timer


def main():
    debug = False
    src_dir = "../data/02_transform/"
    dst_dir = "../data/03_split/"

    if debug:
        num_split = 5
        train_size = 10_000_000
        valid_size = 250_000
        eval_size = 500_000
    else:
        num_split = 5
        train_size = 50_000_000
        valid_size = 2500_000
        eval_size = 5000_000

    # Load data.
    train = cudf.read_csv("../data/raw/train.csv")
    train = train.to_pandas()
    # Preprocessing
    train["prior_question_had_explanation"] = (
        train["prior_question_had_explanation"]
        .map({True: 1, False: 0})
        .fillna(-999)
        .astype(int)
    )
    print(train.head())

    max_timestamp_u = (
        train[["user_id", "timestamp"]].groupby(["user_id"]).agg(["max"]).reset_index()
    )
    max_timestamp_u.columns = ["user_id", "max_timestamp_by_user_id"]
    MAX_TIME_STAMP = max_timestamp_u["max_timestamp_by_user_id"].max()

    def rand_time(max_timestamp_by_user_id):
        interval = MAX_TIME_STAMP - max_timestamp_by_user_id
        rand_time_stamp = random.randint(0, interval)
        return rand_time_stamp

    max_timestamp_u["rand_timestamp"] = max_timestamp_u[
        "max_timestamp_by_user_id"
    ].apply(rand_time)

    train = train.merge(max_timestamp_u, on="user_id", how="left")
    train["viretual_timestamp"] = train["timestamp"] + train["rand_timestamp"]

    train = train[-train_size:]
    train = train.sort_values(["viretual_timestamp", "row_id"]).reset_index(drop=True)

    drop_cols = ["viretual_timestamp", "rand_timestamp", "max_timestamp_by_user_id"]
    features = train.columns.difference(drop_cols)

    # Dump eval dataset.
    eval_dataset = train[-eval_size:]
    train = train[:-eval_size]
    eval_dataset[features].to_pickle(os.path.join(dst_dir, "eval.pkl"))

    for n_fold in range(num_split):
        print(">" * 5, f"Dump files {n_fold} fold.")
        valid = train[-valid_size:]
        train = train[:-valid_size]

        new_users = len(valid[~valid.user_id.isin(train.user_id)].user_id.unique())
        new_contents = len(
            valid[~valid.content_id.isin(train.content_id)].content_id.unique()
        )

        valid[features].to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_valid.pkl"))
        train[features].to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_train.pkl"))

        print(valid.head())
        print(
            "train Unique User: ",
            train["user_id"].nunique(),
            ", train taget mean:",
            train["answered_correctly"].mean(),
        )
        print(
            "valid Unique User: ",
            valid["user_id"].nunique(),
            ", train taget mean:",
            valid["answered_correctly"].mean(),
        )
        print(f"New User Count is {new_users}")
        print(f"New Content Count is {new_contents}")
        print("")


if __name__ == "__main__":
    with timer("Split CV"):
        main()
