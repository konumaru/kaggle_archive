import pickle
from typing import Callable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


def xgb_amex(y_pred, y_true):
    return "amex", amex_metric_mod(y_true.get_label(), y_pred)


def amex_metric_mod(y_true, y_pred):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def train_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "booster": "gbtree",
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 1.5,
        "min_child_weight": 20,
        "eta": 0.05,
        "scale_pos_weight": 2.8,
        # "disable_default_eval_metric": 1,
        "random_state": seed,
    }
    oof_predictions = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(X, y)):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {X.shape[1]} features...")

        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        dtrain = xgb.DMatrix(data=x_train, label=y_train, feature_weights=[7, 2, 1])
        dvalid = xgb.DMatrix(data=x_val, label=y_val)

        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        bst = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=watchlist,
            early_stopping_rounds=50,
            custom_metric=xgb_amex,
            maximize=True,
            verbose_eval=100,
        )
        bst.save_model(f"./output/xgb_seed={seed}_fold={fold}.json")

        pred_val = bst.predict(dvalid, iteration_range=(0, bst.best_ntree_limit))
        oof_predictions[val_ind] = pred_val

    pd.Series(oof_predictions, name="oof").to_csv(
        f"./output/oof_seed={seed}.csv", index=False
    )

    return oof_predictions


def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return "amex_metric", amex_metric_mod(y_true, y_pred), True


def train_lgb(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    params = {
        "objective": "binary",
        # "metric": "binary_logloss",
        "metric": "None",
        "boosting": "dart",
        "seed": seed,
        "num_leaves": 64,
        "learning_rate": 0.05,
        "bagging_fraction": 0.8,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "device_type": "cpu",
        "scale_pos_weight": 2.8,
    }
    oof_predictions = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(X, y)):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {X.shape[1]} features...")

        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
        lgb_valid = lgb.Dataset(x_val, y_val, free_raw_data=False)

        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_valid],
            feval=lgb_amex_metric,
            callbacks=[lgb.log_evaluation(100)],
        )
        with open(f"./output/lgb_seed={seed}_fold={fold}.pickle", "wb") as f:
            pickle.dump(model, f)

        pred_val = model.predict(x_val)
        oof_predictions[val_ind] = pred_val

    pd.Series(oof_predictions, name="oof").to_csv(
        f"./output/oof_seed={seed}.csv", index=False
    )

    return oof_predictions


def train_rsa(
    X: pd.DataFrame,
    y: pd.Series,
    num_seed: int,
    train_fn: Callable,
    seed: int = 42,
):

    rsa_oof_predictions = np.zeros(len(X))
    for i in range(num_seed):
        sub_seed = seed + i
        print(f"\n\n=== Training with seed={sub_seed} ===")

        oof = train_fn(X, y, sub_seed)
        rsa_oof_predictions += oof / num_seed

    pd.Series(rsa_oof_predictions, name="oof").to_csv("./output/oof.csv", index=False)

    return rsa_oof_predictions


def submission(
    ensemble_exp: List, num_fold: int = 5, num_seed: int = 5, seed: int = 42
):
    submissions = []
    for exp_name in ensemble_exp:
        submission = pd.read_csv(f"../{exp_name}/output/submission.csv")["prediction"]
        submission.rename(exp_name, inplace=True)
        submissions.append(submission)
    X = pd.concat(submissions, axis=1)

    predictions = np.zeros(len(X))
    for i in range(num_seed):
        sub_seed = seed + 1
        for fold in tqdm(range(num_fold)):
            bst = xgb.Booster()
            bst.load_model(f"./output/xgb_seed={sub_seed}_fold={fold}.json")
            dtest = xgb.DMatrix(data=X)
            predictions += bst.predict(dtest) / (num_fold * num_seed)

    customer_id = pd.read_parquet("../exp_00109/output/test_fe.parquet")["customer_ID"]
    submission = pd.DataFrame(
        {
            "customer_ID": customer_id,
            "prediction": predictions,
        }
    )
    submission.to_csv("./output/submission.csv", index=False)


def main():
    ensemble_exp = [
        "exp_00110",  # xgb add te feature
        "exp_00200",  # lgbm
        "exp_00300",  # tabnet
    ]

    oofs = []
    for exp_name in ensemble_exp:
        oof = pd.read_csv(f"../{exp_name}/output/oof.csv")
        oof.rename(columns={"oof": exp_name}, inplace=True)
        oofs.append(oof)

    X = pd.concat(oofs, axis=1)
    y = pd.read_parquet("../exp_00109/output/train_fe.parquet")["target"]

    oof = train_rsa(X, y, 5, train_lgb, 42)

    score = amex_metric_mod(y, oof)
    print(f"Our out of folds CV score is {score}")

    with open(f"score={score:.6f}.txt", "w") as f:
        f.write("")

    submission(ensemble_exp, num_fold=5, num_seed=5)


if __name__ == "__main__":
    main()
