import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
from collections import defaultdict
from joblib import Parallel, delayed

from utils.common import dump_pickle

from func import UserState, get_feature_and_update_state


def run_transform(data, is_train=True):
    us = UserState()
    if not is_train:
        us.load_state("../data/04_create_feature/user_state.pkl")
    result = Parallel(n_jobs=-1)(
        delayed(get_feature_and_update_state)(us, idx, row)
        for idx, row in tqdm(data.iterrows(), total=len(data))
    )
    us.dump_state("../data/04_create_feature/user_state.pkl")
    result = pd.DataFrame(result)
    result = result.loc[result["content_type_id"] == 0].reset_index(drop=True)

    X = result.drop(
        ["user_id", "row_id", "timestamp", "content_type_id", "answered_correctly"],
        axis=1,
    )
    X.fillna(-999, inplace=True)
    y = result["answered_correctly"]
    weight = 1 / result["user_id"].map(result["user_id"].value_counts().to_dict())
    return X, y, weight


def main():
    debug = True
    src_dir = "../data/03_split/"
    dst_dir = "../data/04_create_feature/"

    content = pd.read_pickle("../data/02_transform/content.pkl")
    task_container = pd.read_csv("../data/01_preprocessing/task_container_feature.csv")

    # Train and valid dataset created feature.
    num_fold = 5
    for n_fold in range(num_fold):
        print(">" * 5, f"Now processing {n_fold} Fold.")
        trainset = pd.read_pickle(os.path.join(src_dir, f"{n_fold}_fold_train.pkl"))
        validset = pd.read_pickle(os.path.join(src_dir, f"{n_fold}_fold_valid.pkl"))

        if debug:
            trainset = trainset.iloc[-500_000:]
            validset = validset.iloc[-100_000:]

        X_train, y_train, weight_train = run_transform(trainset)
        X_valid, y_valid, weight_valid = run_transform(validset, is_train=False)

        # Merge content features
        X_train = X_train.merge(content, how="left", on="content_id")
        X_train = X_train.merge(task_container, how="left", on="task_container_id")
        X_train.fillna(-999, inplace=True)
        X_valid = X_valid.merge(content, how="left", on="content_id")
        X_valid = X_valid.merge(task_container, how="left", on="task_container_id")
        X_valid.fillna(-999, inplace=True)

        print(X_train.shape)
        print(X_train.head())

        X_train.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_X_train.pkl"))
        y_train.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_y_train.pkl"))
        weight_train.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_weight_train.pkl"))
        X_valid.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_X_valid.pkl"))
        y_valid.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_y_valid.pkl"))
        weight_valid.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_weight_valid.pkl"))

    dump_pickle(X_valid.columns.tolist(), os.path.join(dst_dir, "feature_names.pkl"))

    # Evaluation dataset created feature.
    print(">" * 5, f"Now processing evaluation dataset.")
    evalset = pd.read_pickle("../data/03_split/eval.pkl")
    if debug:
        evalset = evalset.iloc[-500_000:]

    X_eval, y_eval, weight_eval = run_transform(evalset, is_train=False)
    X_eval = X_eval.merge(content, how="left", on="content_id")
    X_eval = X_eval.merge(task_container, how="left", on="task_container_id")
    X_eval.fillna(-999, inplace=True)

    X_eval.to_pickle(os.path.join(dst_dir, "X_eval.pkl"))
    y_eval.to_pickle(os.path.join(dst_dir, "y_eval.pkl"))
    weight_eval.to_pickle(os.path.join(dst_dir, "weight_eval.pkl"))
    # Dump groups
    groups = evalset.loc[evalset["content_type_id"] == 0, "user_id"].reset_index(
        drop=True
    )
    groups.to_pickle(os.path.join(dst_dir, "groups_eval.pkl"))


if __name__ == "__main__":
    main()
