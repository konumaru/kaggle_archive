import os
import copy
import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from utils.common import timer
from utils.common import load_pickle, dump_pickle
from utils.common import append_list_as_row

from utils.metric import fast_auc
from utils.trainer import XGBTrainer
from utils.plot import plot_importance, plot_roc_curve


def dump_results(auc: float):
    """Dump cross validation score to csv file.

    Parameters
    ----------
    auc : float
        score of cross validation.
    """
    running_date = datetime.date.today()
    append_list_as_row("../data/result/cv_score.csv", [running_date, auc])


def train_model(seed: int):
    categorical_cols = [
        "content_id",
        "content_type_id",
        "task_container_id",
    ]

    params = {
        "eta": 0.1,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 1.0,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    params["seed"] = seed
    train_params = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "verbose_eval": 50,
    }

    num_fold = 5
    data_dir = "../data/split/"

    oof = np.array([])
    target = np.array([])
    models = []
    importances = []
    for n_fold in range(num_fold):
        print(f">>>>> {n_fold+1}-Fold")

        X_train = pd.read_pickle(os.path.join(data_dir, f"{n_fold}_fold_X_train.pkl"))
        y_train = pd.read_pickle(os.path.join(data_dir, f"{n_fold}_fold_y_train.pkl"))
        X_valid = pd.read_pickle(os.path.join(data_dir, f"{n_fold}_fold_X_valid.pkl"))
        y_valid = pd.read_pickle(os.path.join(data_dir, f"{n_fold}_fold_y_valid.pkl"))

        weight_train = pd.read_pickle(
            os.path.join(data_dir, f"{n_fold}_fold_weight_train.pkl")
        )
        weight_valid = pd.read_pickle(
            os.path.join(data_dir, f"{n_fold}_fold_weight_valid.pkl")
        )

        train_model = XGBTrainer()
        train_model.fit(
            params,
            train_params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            weight_train=weight_train.to_numpy(),
            weight_valid=weight_valid.to_numpy(),
        )

        fold_pred = train_model.predict(X_valid)
        model = train_model.get_model()
        importance = train_model.get_importance()

        oof = np.concatenate([oof, fold_pred], axis=0)
        target = np.concatenate([target, y_valid.to_numpy()], axis=0)
        models.append(model)
        importances.append(importance)

    auc = fast_auc(target, oof)
    print(f"AUC: {auc}\n")

    return oof, target, models, importances


def run_train():
    num_seed = 1
    seeds = [42 + i for i in range(num_seed)]

    oof = []
    target = []
    models = []
    importances = []
    for seed in seeds:
        print("\n" + "=" * 10, f"seed={seed}", "=" * 10, "\n")
        _oof, _target, _models, _importances = train_model(seed)

        oof.append(_oof)
        target.append(_target)
        models.extend(_models)
        importances.extend(_importances)

    oof = np.mean(oof, axis=0)
    target = np.mean(target, axis=0)

    return oof, target, models, importances


def average_importance(importances: List[Dict], max_feature: int = 50):
    importances = pd.DataFrame(importances).T

    importances = importances.assign(
        mean_feature_importance=importances.mean(axis=1),
        std_feature_importance=importances.std(axis=1),
    )
    importances = importances.sort_values(by="mean_feature_importance")

    if max_feature is not None:
        importances = importances.iloc[:max_feature]

    name = importances.index
    mean_importance = importances["mean_feature_importance"]
    std_importance = importances["std_feature_importance"]
    return name, mean_importance, std_importance


def main():
    dump_dir = "../data/train/"

    oof, target, models, importances = run_train()
    print("Train Completed.")
    # Dump train results.
    dump_pickle(oof, os.path.join(dump_dir, "oof.pkl"))
    dump_pickle(target, os.path.join(dump_dir, "target.pkl"))
    dump_pickle(models, os.path.join(dump_dir, "models.pkl"))
    dump_pickle(importances, os.path.join(dump_dir, "importances.pkl"))
    # Load train results.
    oof = load_pickle(os.path.join(dump_dir, "oof.pkl"))
    target = load_pickle(os.path.join(dump_dir, "target.pkl"))
    models = load_pickle(os.path.join(dump_dir, "models.pkl"))
    importances = load_pickle(os.path.join(dump_dir, "importances.pkl"))

    auc = fast_auc(target, oof)
    print(f"AUC: {auc}")

    name, avg_importance, std_importance = average_importance(importances)
    save_path = os.path.join(dump_dir, "importance.png")
    plot_importance(save_path, y=name, data=avg_importance, xerr=std_importance)

    save_path = os.path.join(dump_dir, "roc_curve.png")
    plot_roc_curve(target, oof, save_path)
    # Dump cv score to csv file.
    dump_results(auc)


if __name__ == "__main__":
    with timer("Train"):
        main()


"""
# Parameters for LightGBM
params = {
    "objective": "binary",
    "learning_rate": 0.3,
    "max_depth": 7,
    "num_leaves": 2 ** 7,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 20,
    "bagging_fraction": 1.0,
    "metric": "auc",
    "force_row_wise": False,
    "force_col_wise": True,
    "verbosity": -1,
}
train_params = {
    "num_boost_round": 1000,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,
}
"""
