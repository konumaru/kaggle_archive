import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


def xgb_amex(y_pred, y_true):
    return "amex", amex_metric_np(y_pred, y_true.get_label())


# Created by https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max
    return 0.5 * (g + d)


def main():
    train = pd.read_parquet("./output/train_fe.parquet")
    features = [col for col in train.columns if col not in ["customer_ID", "target"]]

    params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "max_depth": 7,
        "subsample": 0.88,
        "colsample_bytree": 0.5,
        "gamma": 1.5,
        "min_child_weight": 8,
        "lambda": 70,
        "eta": 0.03,
    }
    oof_predictions = np.zeros(len(train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (trn_ind, val_ind) in enumerate(cv.split(train, train["target"])):
        print("")
        print("-" * 50)
        print(f"Training fold {fold} with {len(features)} features...")

        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train["target"].iloc[trn_ind], train["target"].iloc[val_ind]

        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dvalid = xgb.DMatrix(data=x_val, label=y_val)

        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        bst = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=2600,
            evals=watchlist,
            early_stopping_rounds=500,
            custom_metric=xgb_amex,
            maximize=True,
            verbose_eval=200,
        )
        bst.save_model(f"./output/xgb_{fold}.json")

        pred_val = bst.predict(dvalid, iteration_range=(0, bst.best_ntree_limit))
        oof_predictions[val_ind] = pred_val

    # NOTE: Compute out of folds metric.
    oof_predictions = pd.Series(oof_predictions, name="oof")
    oof_predictions.to_csv("./output/oof.csv", index=False)

    score = amex_metric_np(oof_predictions.to_numpy(), train["target"].to_numpy())

    print(f"Our out of folds CV score is {score}")
    with open(f"score={score:.6f}.txt", "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
