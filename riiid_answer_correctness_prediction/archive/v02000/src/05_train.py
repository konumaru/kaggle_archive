import os
import gc
import copy
import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

import torch

from utils.common import timer
from utils.common import load_pickle, dump_pickle

from utils.metric import fast_auc
from utils.trainer.tabnet import TabNetClassificationTrainer
from utils.trainer.gbdt import XGBTrainer, LGBTrainer
from utils.plot import plot_importance, plot_roc_curve


def train_model(Trainer, params, train_params, num_fold: int = 5):
    categorical_cols = [
        "content_id",
    ]

    srd_dir = "../data/04_create_feature/"

    oof = np.array([])
    target = np.array([])
    models = []
    importances = []
    for n_fold in range(num_fold):
        print(f">>>>> {n_fold}-Fold")

        X_train = pd.read_pickle(os.path.join(srd_dir, f"{n_fold}_fold_X_train.pkl"))
        y_train = pd.read_pickle(os.path.join(srd_dir, f"{n_fold}_fold_y_train.pkl"))
        X_valid = pd.read_pickle(os.path.join(srd_dir, f"{n_fold}_fold_X_valid.pkl"))
        y_valid = pd.read_pickle(os.path.join(srd_dir, f"{n_fold}_fold_y_valid.pkl"))

        weight_train = pd.read_pickle(
            os.path.join(srd_dir, f"{n_fold}_fold_weight_train.pkl")
        )
        weight_valid = pd.read_pickle(
            os.path.join(srd_dir, f"{n_fold}_fold_weight_valid.pkl")
        )

        trainer = Trainer()
        trainer.fit(
            params,
            train_params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            # weight_train=weight_train.to_numpy(),
            # weight_valid=weight_valid.to_numpy(),
        )

        fold_pred = trainer.predict(X_valid)
        model = trainer.get_model()
        importance = trainer.get_importance()

        oof = np.concatenate([oof, fold_pred], axis=0)
        target = np.concatenate([target, y_valid.to_numpy()], axis=0)
        models.append(model)
        importances.append(importance)

    auc = fast_auc(target, oof)
    print(f"AUC: {auc}\n")

    return oof, target, models, importances


def run_train():
    srd_dir = "../data/04_create_feature/"
    dst_dir = "../data/05_train/"

    num_seed = 1  # 5
    seeds = [42 + i for i in range(num_seed)]
    trainers = {
        "xgb": {
            "trainer": XGBTrainer,
            "params": {
                "eta": 0.1,
                "max_depth": 9,
                # "min_child_weight": 20,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "objective": "binary:logistic",
                "eval_metric": "auc",
                # "gpu_id": 0,
                "seed": None,
            },
            "train_params": {
                "num_boost_round": 1000,
                "early_stopping_rounds": 20,
                "verbose_eval": 50,
            },
        },
        "lgbm": {
            "trainer": LGBTrainer,
            "params": {
                "objective": "binary",
                "learning_rate": 0.1,
                "max_depth": -1,
                "num_leaves": 2 ** 7,
                "max_bin": 700,
                "min_data_in_leaf": 20,
                "min_sum_hessian_in_leaf": 20,
                "feature_fraction": 0.6,
                "bagging_freq": 1,
                "bagging_fraction": 0.6,
                "bagging_seed": 11,
                "metric": "auc",
                "force_row_wise": False,
                "force_col_wise": True,
                "verbosity": -1,
                "seed": None,
            },
            "train_params": {
                "num_boost_round": 1000,
                "early_stopping_rounds": 50,
                "verbose_eval": 50,
            },
        },
        # NOTE:
        # - 学習が遅すぎるので今回は不採用
        # - また、このままのコードではGPUのメモリ不足で実行負荷なので改善が必要
        # "tabnet": {
        #     "trainer": TabNetClassificationTrainer,
        #     "params": {
        #         "n_d": 16,
        #         "n_a": 16,
        #         "n_steps": 3,
        #         "cat_idxs": [0],
        #         "seed": None,
        #         "device_name": "cuda:0",
        #     },
        #     "train_params": {
        #         "eval_metric": ["auc"],
        #         "max_epochs": 1,  # 500,
        #         "patience": 20,
        #         "batch_size": 512,
        #         "num_workers": 4,
        #         "weights": 1,
        #         "drop_last": True,
        #     },
        # },
    }

    oof_results = {}
    for trainer_name, trainer_dict in trainers.items():
        trainer = trainer_dict["trainer"]
        params = trainer_dict["params"]
        train_params = trainer_dict["train_params"]

        oof = []
        target = []
        models = []
        importances = []
        for seed in seeds:
            print("#" * 25)
            print("#" * 5, f"Seed {seed}")
            print("#" * 25)
            params["seed"] = seed
            _oof, _target, _models, _importances = train_model(
                trainer, params, train_params
            )
            oof.append(_oof)
            target.append(_target)
            models.extend(_models)
            importances.extend(_importances)

        for seed, _oof in zip(seeds, oof):
            auc = fast_auc(target[0], _oof)
            print(f"Seed {seed}: AUC is {auc:.6f}")

        avg_oof = np.mean(oof, axis=0)
        auc = fast_auc(target[0], avg_oof)
        print(f"Averaging Score: AUC is {auc:.6f}")

        # Dump train results.
        dump_pickle(avg_oof, os.path.join(dst_dir, f"{trainer_name}_oof.pkl"))
        dump_pickle(target[0], os.path.join(dst_dir, f"{trainer_name}_target.pkl"))
        dump_pickle(models, os.path.join(dst_dir, f"{trainer_name}_models.pkl"))
        dump_pickle(
            importances, os.path.join(dst_dir, f"{trainer_name}_importances.pkl")
        )

        name, avg_importance, std_importance = average_importance(importances)
        save_path = os.path.join(dst_dir, f"{trainer_name}_importance.png")
        plot_importance(save_path, y=name, data=avg_importance, xerr=std_importance)

        save_path = os.path.join(dst_dir, f"{trainer_name}_roc_curve.png")
        plot_roc_curve(target[0], avg_oof, save_path)

        oof_results[trainer_name] = avg_oof

    ensenble_oof = np.mean(list(oof_results.values()), axis=0)

    save_path = os.path.join(dst_dir, f"ensemble_roc_curve.png")
    plot_roc_curve(target[0], ensenble_oof, save_path)

    auc = fast_auc(target[0], avg_oof)
    print(f"Ensemble Score: AUC is {auc:.6f}")


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
    run_train()


if __name__ == "__main__":
    with timer("Train"):
        main()
