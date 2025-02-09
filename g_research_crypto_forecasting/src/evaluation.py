import os
import pathlib
import pickle
from operator import is_not

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

plt.style.use("seaborn-darkgrid")

SEED = 42
NUM_FOLD = 3
NUM_SEED = 3

data_dir = pathlib.Path("../data/")


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def predict(data: pd.DataFrame, model_name: str):
    model_dir = data_dir / f"model/{model_name}/"

    if model_name == "xgb":
        data = xgboost.DMatrix(data)

    preds = []
    for n_fold in range(NUM_FOLD):
        for i in range(NUM_SEED):
            seed = SEED + i
            model = load_pickle(model_dir / f"seed{seed:03}_fold{n_fold:02}.pkl")
            pred = model.predict(data)
            preds.append(pred)
    pred_avg = np.mean(preds, axis=0)
    return pred_avg


def weighted_correlation(a: np.ndarray, b: np.ndarray, weights: np.ndarray):
    """The evaluation metric is weighted correlation as opposed to a weighted mean of correlation.
    The metric is defined as follows, where 'a', 'b' and 'weights' are vectors of the same length.
    'a' and 'b' are the expected and predicted targets, and ' weights' include the weight of each row, determined by its asset:

        Args:
            a ([np.ndarray]): [description]
            b ([np.ndarray]): [description]
            weights ([np.ndarray]): [description]

        Returns:
            corr [float]: weighted correlation
    """
    w = np.ravel(weights)
    a = np.ravel(a)
    b = np.ravel(b)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)
    return corr


def evaluate(y_true, y_pred, weights=None):
    assert y_true.shape == y_pred.shape
    error = mean_squared_error(y_true, y_pred)
    if weights is not None:
        metric = weighted_correlation(y_true, y_pred, weights)
    else:
        metric = pearsonr(y_true, y_pred)[0]
    return error, metric


def savefig_each_asset_mse(result: pd.DataFrame):
    plt.figure()
    result.plot(kind="bar", x="Asset_Name", y="MSE")
    plt.ylim(0.0, None)
    plt.ylabel("mse")
    plt.tight_layout()
    plt.savefig("../data/result/mse.png")


def savefig_each_asset_wcorr(result: pd.DataFrame):
    plt.figure()
    result.plot(kind="bar", x="Asset_Name", y="Pearson_Corr")
    plt.ylim(-0.05, 0.1)
    plt.ylabel("pearson corr")
    plt.tight_layout()
    plt.savefig("../data/result/wcorr.png")


def savefig_pred_target_scatter(pred, target, c=None):
    plt.figure()
    plt.scatter(pred, target, alpha=0.6, c=c)
    plt.xlabel("predict")
    plt.ylabel("target")
    plt.tight_layout()
    if c is not None:
        plt.savefig("../data/result/scatter_each_asset.png")
    else:
        plt.savefig("../data/result/scatter.png")


def main():
    os.makedirs("../data/result", exist_ok=True)

    asset_details = pd.read_csv(data_dir / "raw/asset_details.csv")
    test = pd.read_pickle(data_dir / "split/test.pkl")
    X_test = pd.read_pickle(data_dir / "feature/test.pkl")
    target = pd.read_pickle(data_dir / "split/test_target.pkl")
    weight = pd.read_pickle(data_dir / "split/test_weight.pkl")

    is_not_nan = ~target.isnull()
    test = test[is_not_nan]
    X_test = X_test[is_not_nan]
    target = target[is_not_nan]
    weight = weight[is_not_nan]

    all_pred = []
    for model_name in ["lgb", "xgb"]:
        pred = predict(X_test, model_name)
        all_pred.append(pred)
    all_pred_avg = np.mean(all_pred, axis=0)

    result = []
    error, metric = evaluate(target, all_pred_avg, weight)
    result.append(("Overall", error, metric))
    for _, row in asset_details.sort_values(by="Asset_ID").iterrows():
        is_asset_row = X_test["Asset_ID"] == row["Asset_ID"]

        _target = target[is_asset_row]
        pred = all_pred_avg[is_asset_row]
        error, metric = evaluate(_target, pred, weights=None)

        result.append((row["Asset_Name"], error, metric))

    result = pd.DataFrame(result, columns=["Asset_Name", "MSE", "Pearson_Corr"])
    print(result)
    savefig_each_asset_mse(result)
    savefig_each_asset_wcorr(result)
    savefig_pred_target_scatter(all_pred_avg, target)
    savefig_pred_target_scatter(all_pred_avg, target, c=test["Asset_ID"])
    result.to_csv(data_dir / "result/result.csv")


if __name__ == "__main__":
    main()
