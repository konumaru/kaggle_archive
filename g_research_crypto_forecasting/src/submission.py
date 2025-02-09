import glob
import pickle
from operator import sub
from random import sample
from typing import List

import numpy as np
import pandas as pd
import xgboost

from create_feature import make_feature
from store import CryptoStore


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_models(files):
    models = [load_pickle(f) for f in files]
    return models


def predict(data: pd.DataFrame, models: List):
    pred = [m.predict(data) for m in models]
    pred = np.mean(pred, axis=0)
    return pred


def main():
    test = pd.read_csv("../data/raw/example_test.csv")
    sample_submission = pd.read_csv("../data/raw/example_sample_submission.csv")

    iter_test = test.groupby("group_num")

    lgb_models = load_models(glob.glob("../data/model/lgb/*.pkl"))
    xgb_models = load_models(glob.glob("../data/model/xgb/*.pkl"))
    store = CryptoStore()

    submissions = []
    for group_num, test_df in iter_test:
        sample_prediction_df = sample_submission.loc[
            sample_submission["group_num"] == group_num, ["row_id", "Target"]
        ]
        test_df.drop("group_num", axis=1, inplace=True)
        store.update(test_df)
        features = make_feature(test_df, store)
        sample_prediction_df["Target"] = predict(features, lgb_models) / 2
        sample_prediction_df["Target"] += (
            predict(xgboost.DMatrix(features), xgb_models) / 2
        )

        submissions.append(sample_prediction_df)

    submission = pd.concat(submissions, axis=0)
    submission.to_csv("../data/submission.csv")
    print(submission.head())


if __name__ == "__main__":
    main()
