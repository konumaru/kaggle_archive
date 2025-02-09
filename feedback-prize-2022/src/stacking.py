import pathlib
import pickle
from typing import List

import numpy as np
import pandas as pd
from cuml import SVR
from cuml.ensemble import RandomForestRegressor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error


def dump_pickle(data, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def split_fold(X: pd.DataFrame, y: pd.DataFrame, num_fold: int = 4, seed: int = 42):
    X = X.assign(fold=0)

    cv = MultilabelStratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(cv.split(X, y)):
        X.loc[valid_idx, "fold"] = fold
    return X


def compute_MCRMSE(y_true: np.ndarray, y_pred: np.ndarray):
    col_rmse = [
        mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        for i in range(y_true.shape[1])
    ]
    return np.mean(col_rmse)


def fit(model_name, model, params, data, feature_cols, target_cols, num_fold=25):
    preds = np.zeros((len(data), len(target_cols)))
    models = {}

    for i, t in enumerate(target_cols):
        trained_models = []

        for fold in range(num_fold):
            train = data[data["fold"] != fold]
            valid = data[data["fold"] == fold]

            clf = model(**params)
            clf.fit(train[feature_cols], train[t])
            trained_models.append(clf)

            preds[data["fold"] == fold, i] = clf.predict(valid[feature_cols])

        models[t] = trained_models

    save_dir = pathlib.Path(f"../data/working/stacking/models_{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    for t_name, t_models in models.items():
        for i, m in enumerate(t_models):
            dump_pickle(m, str(save_dir / f"{t_name}_{i}.pkl"))

    return preds


def main():
    work_dir = pathlib.Path("../data/working/stacking")
    work_dir.mkdir(exist_ok=True)

    num_fold = 25

    train = pd.read_csv("../data/raw/train.csv")
    target_cols = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
    y = train[target_cols].copy()

    ensemble_models = [
        # "deberta-base-256",
        "deberta-base-768",
        "deberta-base",
        "deberta-large-256",
        "deberta-large",
        "deberta-v3-base-256",
        "deberta-v3-base-768",
        "deberta-v3-base",
        "deberta-v3-large-256",
        "deberta-v3-large",
        "roberta-base-256",
        "roberta-base",
        # "roberta-large",
    ]
    oof = []
    for m in ensemble_models:
        _oof = pd.read_csv(f"../data/working/{m}/oof.csv")
        _oof = _oof.set_index("text_id")
        _oof.rename(columns={t: f"{t}_{m}" for t in target_cols}, inplace=True)
        oof.append(_oof.astype(np.float32))

    oof = pd.concat(oof, axis=1).reset_index(drop=True)
    oof = split_fold(oof, y, num_fold=num_fold)
    feature_cols = [c for c in oof.columns if c not in ["text_id", "fold"]]
    data = pd.concat([oof, y], axis=1).reset_index(drop=True)

    preds_svr = fit(
        "svr",
        SVR,
        {"C": 4e-1},
        data,
        feature_cols,
        target_cols,
        num_fold,
    )
    score = compute_MCRMSE(y.to_numpy(), preds_svr)
    with open(str(work_dir / f"score_of_svr={score:.6f}"), "w") as f:
        f.write("")

    preds_rf = fit(
        "rf",
        RandomForestRegressor,
        {
            "n_estimators": 200,
            "max_depth": 5,
            "max_samples": 0.9,
            "min_samples_split": 20,
            "accuracy_metric": "mean_ae",
        },
        data,
        feature_cols,
        target_cols,
        num_fold,
    )
    score = compute_MCRMSE(y.to_numpy(), preds_rf)
    with open(str(work_dir / f"score_of_rf={score:.6f}"), "w") as f:
        f.write("")

    preds = (preds_svr + preds_rf) / 2.0
    preds = np.clip(preds, 1.0, 5.0)
    print(preds)

    score = compute_MCRMSE(y.to_numpy(), preds)
    print(score)
    with open(str(work_dir / f"score={score:.6f}"), "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
