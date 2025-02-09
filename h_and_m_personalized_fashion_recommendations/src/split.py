import os
import pathlib
from datetime import timedelta

import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class TimeSeriesDaliySplit(_BaseKFold):
    """Time Series Daily cross-validator"""

    def __init__(
        self,
        n_splits: int,
        test_days: int,
        max_train_days: int,
    ):
        self.n_splits = n_splits
        self.test_days = test_days
        self.max_train_days = max_train_days

    def split(self, X, y=None, groups=None):
        n_splits = self.get_n_splits()
        max_days = self.max_train_days + self.test_days
        indices = X.index
        latest_date = X["t_dat"].max()

        test_end_date = [latest_date - timedelta(days=i * 7) for i in range(n_splits)]
        train_end_date = [d - timedelta(days=7) for d in test_end_date]

        for _train_end_date, _test_end_date in zip(train_end_date, test_end_date):
            _train_start_date = _train_end_date - timedelta(days=max_days)
            _test_start_date = _test_end_date - timedelta(days=max_days)

            train_indices = indices[
                (X["t_dat"] <= _train_end_date) & (X["t_dat"] > _train_start_date)
            ]
            test_indices = indices[
                (X["t_dat"] <= _test_end_date) & (X["t_dat"] > _test_start_date)
            ]
            yield train_indices, test_indices


def main():
    data = pd.read_csv("../data/raw/transactions_train.csv", dtype={"article_id": str})
    data["t_dat"] = pd.to_datetime(data["t_dat"], format="%Y-%m-%d")
    print(data.tail(), "\n")

    dump_dir = pathlib.Path("../data/split")
    os.makedirs(dump_dir, exist_ok=True)

    print("Create each fold data.")
    num_fold = 5
    num_eval = 1
    num_splits = num_fold + num_eval
    max_train_days = 7 * 5
    cv = TimeSeriesDaliySplit(
        n_splits=num_splits, test_days=7, max_train_days=max_train_days
    )
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(data)):
        if n_fold == 0:
            f_name = "evaluation"
            eval_data = data.iloc[valid_idx]
            print(f"{f_name.title()} >>>")
            eval_data.to_parquet(
                dump_dir / f"{f_name}.parquet.gzip", compression="gzip"
            )
            print("Evaluate shape :", eval_data.shape)
            print(eval_data["t_dat"].min(), "~", eval_data["t_dat"].max())
            print(eval_data.tail())
            print()
        else:
            f_name = n_fold - 1
            train_data = data.iloc[train_idx]
            valid_data = data.iloc[valid_idx]
            print(f"Fold {f_name} >>>")

            train_data.to_parquet(
                dump_dir / f"{f_name}_train.parquet.gzip", compression="gzip"
            )
            valid_data.to_parquet(
                dump_dir / f"{f_name}_valid.parquet.gzip", compression="gzip"
            )

            print("Train shape :", train_data.shape)
            print(train_data["t_dat"].min(), "~", train_data["t_dat"].max())
            print(train_data.head())
            print("Valid shape :", valid_data.shape)
            print(valid_data["t_dat"].min(), "~", valid_data["t_dat"].max())
            print(valid_data.head())
            print()

    print("Create data for submission.")
    submit_end_data = data["t_dat"].max()
    submit_start_date = submit_end_data - timedelta(days=max_train_days)
    submit_data = data.loc[
        (data["t_dat"] <= submit_end_data) & (data["t_dat"] > submit_start_date), :
    ]
    print("Submit shape :", submit_data.shape)
    print(submit_data["t_dat"].min(), "~", submit_data["t_dat"].max())
    print(submit_data.head())
    submit_data.to_parquet("../data/split/submission.parquet.gzip", compression="gzip")


if __name__ == "__main__":
    main()
