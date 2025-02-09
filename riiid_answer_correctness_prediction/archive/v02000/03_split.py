import os
import pandas as pd

from sklearn import model_selection

from utils.common import timer


def main():
    src_dir = "../data/02_transform/"
    dst_dir = "../data/03_split/"

    # Load data.
    train = pd.read_pickle(os.path.join(src_dir, "train.pkl"))
    print(train.shape)
    print(train.head())

    X = train.drop(["answered_correctly"], axis=1).to_numpy()
    y = train["answered_correctly"].to_numpy()
    group = train["user_id"].to_numpy()

    # Split validation
    cv = model_selection.GroupKFold(n_splits=5)
    # cv = model_selection.StratifiedKFold(n_splits=5)
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, group)):
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[valid_idx]

        fold_train.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_train.pkl"))
        fold_valid.to_pickle(os.path.join(dst_dir, f"{n_fold}_fold_valid.pkl"))

        print(">" * 5, f"Dump files {n_fold} fold.")
        print(fold_train.head())
        print(
            "train Unique User: ",
            fold_train["user_id"].nunique(),
            ", train taget mean:",
            y[train_idx].mean(),
        )
        print(
            "valid Unique User: ",
            fold_valid["user_id"].nunique(),
            ", train taget mean:",
            y[valid_idx].mean(),
        )
        print("")


if __name__ == "__main__":
    with timer("Split CV"):
        main()
