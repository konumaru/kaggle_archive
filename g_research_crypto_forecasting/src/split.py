import os
import pathlib
import time
from ast import dump
from datetime import datetime
from re import L

import numpy as np
import pandas as pd

data_dir = pathlib.Path("../data/")


def totimestamp(s: str):
    return np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))


def split_data(data: pd.DataFrame):
    os.makedirs(data_dir / "split/", exist_ok=True)

    test_date = ("01/07/2021", "21/09/2021")
    # split_date = [
    #     [("01/10/2020", "31/03/2021"), ("01/04/2021", "30/06/2021")],
    #     [("01/07/2020", "31/12/2020"), ("01/01/2021", "31/03/2021")],
    #     [("01/04/2020", "30/09/2020"), ("01/10/2020", "31/12/2020")],
    #     [("01/01/2020", "30/06/2020"), ("01/07/2020", "30/09/2020")],
    #     [("01/10/2019", "31/03/2020"), ("01/04/2020", "30/06/2020")],
    # ]

    split_date = [
        [("01/01/2021", "31/03/2021"), ("01/04/2021", "30/06/2021")],
        [("01/10/2020", "31/12/2020"), ("01/01/2021", "31/03/2021")],
        [("01/07/2020", "30/09/2020"), ("01/10/2020", "31/12/2020")],
        [("01/04/2020", "30/06/2020"), ("01/07/2020", "30/09/2020")],
        [("01/01/2020", "31/03/2020"), ("01/04/2020", "30/06/2020")],
    ]

    is_test = (data["timestamp"] > totimestamp(test_date[0])) & (
        data["timestamp"] <= totimestamp(test_date[1])
    )
    test = data[is_test].reset_index(drop=True).copy()
    test.drop(["Target", "Weight"], axis=1).to_pickle(data_dir / "split/test.pkl")
    test["Target"].to_pickle(data_dir / "split/test_target.pkl")
    test["Weight"].to_pickle(data_dir / "split/test_weight.pkl")

    print("Test Size is", test.shape[0])

    for n_fold, (train_date, valid_date) in enumerate(split_date):
        is_train = (data["timestamp"] > totimestamp(train_date[0])) & (
            data["timestamp"] <= totimestamp(train_date[1])
        )
        is_valid = (data["timestamp"] > totimestamp(valid_date[0])) & (
            data["timestamp"] <= totimestamp(valid_date[1])
        )

        train = data[is_train].reset_index(drop=True).copy()
        valid = data[is_valid].reset_index(drop=True).copy()

        def dump_data(data, name: str):
            data.drop(["Target", "Weight"], axis=1).to_pickle(
                data_dir / f"split/fold{n_fold:02}_{name}.pkl"
            )
            data["Target"].to_pickle(
                data_dir / f"split/fold{n_fold:02}_{name}_target.pkl"
            )
            data["Weight"].to_pickle(
                data_dir / f"split/fold{n_fold:02}_{name}_weight.pkl"
            )

        dump_data(train, "train")
        dump_data(valid, "valid")

        print(f"{n_fold} Fold: Train size {len(train)} Valid size {len(valid)}")


def clip_target(df):
    for i in range(14):
        is_asset = df["Asset_ID"] == i
        _target = df.loc[is_asset, "Target"].copy()
        min_target, max_target = _target.quantile(0.05), _target.quantile(0.95)
        df.loc[is_asset, "Target"] = _target.clip(min_target, max_target).to_numpy()
    return df


def main():
    train = pd.read_csv(
        data_dir / "raw/train.csv",
        dtype={
            "Asset_ID": np.int8,
            "Count": np.int32,
            "Open": np.float64,
            "High": np.float64,
            "Low": np.float64,
            "Close": np.float64,
            "Volume": np.float64,
            "VWAP": np.float64,
            "Target": np.float64,
        },
    )
    asset_details = pd.read_csv(data_dir / "raw/asset_details.csv")

    train = clip_target(train)

    # NOTE: マーケットのデータは使える可能性があるのでpivotしたあとに捨てるほうが好ましい
    # train = train.dropna(subset=["Target"], axis=0)
    # for i in range(14):
    #     _target = train.loc[train["Asset_ID"] == i, "Target"].copy()
    #     train.loc[train["Asset_ID"] == i, "Target"] = _target.clip(
    #         _target.quantile(0.1), _target.quantile(0.9)
    #     )

    print(train.head())
    print(asset_details)
    print()
    # Add weight columns for each asset_ID.
    assetId_weight = {
        row["Asset_ID"]: row["Weight"]
        for _, row in asset_details[["Asset_ID", "Weight"]].iterrows()
    }
    train["Weight"] = train["Asset_ID"].map(assetId_weight)

    split_data(train)


if __name__ == "__main__":
    main()
