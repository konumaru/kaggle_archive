import os
import datetime
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

from utils.metric import fast_auc
from utils.common import load_pickle, dump_pickle
from utils.plot import plot_importance, plot_roc_curve
from utils.common import append_list_as_row  # TODO: 名前変えたほうがいい


def dump_results(auc: float):
    """Dump cross validation score to csv file.

    Parameters
    ----------
    auc : float
        score of cross validation.
    """
    running_date = datetime.date.today()
    append_list_as_row("../data/06_evaluation/cv_score.csv", [running_date, auc])


def predict_xgb(models, data):
    pred = np.mean(
        [m.predict(xgb.DMatrix(data), ntree_limit=m.best_ntree_limit) for m in models],
        axis=0,
    )
    return pred


def predict_lgbm(models, data):
    pred = np.mean(
        [m.predict(data, num_iteration=m.best_iteration) for m in models], axis=0
    )
    return pred


def predict_tabbet(models, data):
    pred = np.mean([m.predict(data.to_numpy()) for m in models], axis=0)
    return pred


def main():
    src_dir = "../data/04_create_feature/"
    dst_dir = "../data/06_evaluation/"
    # Load data.
    X = pd.read_pickle(os.path.join(src_dir, "X_eval.pkl"))
    y = pd.read_pickle(os.path.join(src_dir, "y_eval.pkl"))
    feature_names = load_pickle(os.path.join(src_dir, "feature_names.pkl"))
    # Load models.
    X = X[feature_names]
    pred = np.mean(
        [
            predict_xgb(load_pickle("../data/05_train/xgb_models.pkl"), X),
            predict_lgbm(load_pickle("../data/05_train/lgbm_models.pkl"), X),
            # predict_tabbet(load_pickle("../data/05_train/tabnet_models.pkl"), X),
        ],
        axis=0,
    )

    save_path = os.path.join(dst_dir, f"evaluation_dataset_roc_curve.png")
    plot_roc_curve(y.to_numpy(), pred, save_path)

    auc = fast_auc(y.to_numpy(), pred)
    print(f"Ensemble Score: AUC is {auc:.6f}")
    dump_results(auc)


if __name__ == "__main__":
    main()
