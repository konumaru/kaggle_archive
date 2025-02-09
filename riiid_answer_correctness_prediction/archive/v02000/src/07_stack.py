import os
import numpy as np
import pandas as pd

from sklearn import model_selection

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from utils.metric import fast_auc
from utils.common import load_pickle, dump_pickle
from utils.plot import plot_importance, plot_roc_curve
from utils.plot import plot_confusion_matrix

from utils.trainer.gbdt import XGBTrainer


def plot_corr_heatmap(corr_data: pd.DataFrame, filepath: str):
    plt.figure()
    sns.heatmap(corr_data, annot=True, fmt="1.4f", cmap="Blues", linewidths=0.5)
    plt.savefig(filepath)


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


class CVTrainer(object):
    def __init__(self, trainer):
        self.base_trainer = trainer
        self.oof = None
        self.importance = []
        self.models = []

    def fit(self, params, train_params, cv, X, y, weight=None, groups=None):
        oof = np.zeros(len(y))
        models = []
        importances = []

        for n_fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=groups)):
            print(">" * 5, f"{n_fold} Fold.")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            # weight_train, weight_valid = weight.iloc[train_idx], weight.iloc[valid_idx]

            trainer = self.base_trainer()
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

            oof[valid_idx] = trainer.predict(X_valid)
            models.append(trainer.get_model())
            importances.append(trainer.get_importance())

        self.oof = oof
        self.models = models
        self.importance = importances

    def predict(self, data):
        pass

    def get_models(self):
        return self.models

    def get_importance(self, max_feature: int = 50):
        importances = pd.DataFrame(self.importance).T
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
    src_dir = "../data/05_train/"
    dst_dir = "../data/07_stack/"

    X = pd.read_pickle("../data/04_create_feature/X_eval.pkl")
    y = pd.read_pickle("../data/04_create_feature/y_eval.pkl")
    # weight = pd.read_pickle("../data/04_create_feature/weight_eval.pkl")
    groups = pd.read_pickle("../data/04_create_feature/groups_eval.pkl")
    feature_names = load_pickle("../data/04_create_feature/feature_names.pkl")

    xgb_pred = predict_xgb(load_pickle("../data/05_train/xgb_models.pkl"), X)
    lgb_pred = predict_lgbm(load_pickle("../data/05_train/lgbm_models.pkl"), X)

    pred = np.array([xgb_pred, lgb_pred]).T
    X_preds = pd.DataFrame(pred, columns=["xgb", "lgb"])

    # Save oof heatmap.
    save_path = os.path.join(dst_dir, "corr_heatmap.png")
    plot_corr_heatmap(X_preds.corr(), save_path)

    params = {
        "eta": 0.2,
        "max_depth": 3,
        "min_child_weight": 50,
        "subsample": 0.8,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
    }
    train_params = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 20,
        "verbose_eval": 50,
    }

    # cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    cv = model_selection.GroupKFold(n_splits=5)
    cv_trainer = CVTrainer(trainer=XGBTrainer)
    cv_trainer.fit(params, train_params, cv, X_preds, y, weight=None, groups=groups)

    # Save models.
    dump_pickle(cv_trainer.models, os.path.join(dst_dir, f"stack_models.pkl"))

    # Stacking score.
    auc = fast_auc(y.to_numpy(), cv_trainer.oof)
    print(f"AUC: {auc}\n")
    save_path = os.path.join(dst_dir, "stack_roc_curve.png")
    plot_roc_curve(y.to_numpy(), cv_trainer.oof, save_path)

    # Save importance.
    name, avg_importance, std_importance = cv_trainer.get_importance()
    save_path = os.path.join(dst_dir, "stack_importance.png")
    plot_importance(save_path, y=name, data=avg_importance, xerr=std_importance)


if __name__ == "__main__":
    main()
