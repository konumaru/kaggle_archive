import os

import cudf
import numpy as np
import pandas as pd

import xgboost as xgb

from utils.metric import fast_auc
from utils.common import load_pickle
from utils.plot import plot_roc_curve


def xgb_predictor(model, data):
    pred = model.predict(xgb.DMatrix(data), ntree_limit=model.best_ntree_limit)
    return pred


def main():
    eval_dataset = pd.read_csv("../data/eval_dataset/data.csv")
    eval_dataset = eval_dataset.fillna(-1)

    feature_names = load_pickle("../data/split/feature_names.pkl")
    models = load_pickle("../data/train/models.pkl")

    X = eval_dataset[feature_names]
    y = eval_dataset["answered_correctly"].to_numpy()

    preds = np.zeros(y.shape[0])
    for model in models:
        preds += xgb_predictor(model, X) / len(models)

    auc = fast_auc(y, preds)
    print(f"AUC: {auc}")

    dump_dir = "../data/evaluation/"
    save_path = os.path.join(dump_dir, "roc_curve.png")
    plot_roc_curve(y, preds, save_path)


if __name__ == "__main__":
    main()
