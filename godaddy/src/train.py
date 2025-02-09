import pathlib
from typing import Any

import hydra
import numpy as np
import xgboost as xgb
from rich.progress import track
from sklearn.ensemble import RandomForestRegressor

from config import Config
from metric import smape
from postprocessing import fill_last_mbd
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle


def get_model(model_name: str = "rf", seed: int = 42) -> Any:
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=seed,
        )
    elif model_name == "xgb":
        return xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            tree_method="hist",
            n_estimators=100,  # 795,
            learning_rate=0.2,
            max_leaves=17,
            subsample=0.8,
            max_bin=4096,
            reg_lambda=10,
            n_jobs=4,
            seed=seed,
        )
    else:
        return None


def train(
    X_train,
    y_train,
    gruop_train,
    X_valid=None,
    y_valid=None,
    group_valid=None,
    model_name="rf",
    seed=42,
) -> None:
    model_dir = pathlib.Path(f"./data/model/{model_name}/seed={seed}")
    model_dir.mkdir(exist_ok=True)

    for cfips in track(np.unique(gruop_train)):
        _X_train = X_train[gruop_train == cfips]
        _y_train = y_train[gruop_train == cfips]

        model = get_model(model_name, seed)
        model.fit(_X_train, _y_train)
        save_pickle(str(model_dir / f"{cfips}.pkl"), model)


def predict(X_valid, y_valid, group_valid, model_name="rf") -> np.ndarray:
    model_dir = pathlib.Path(f"./data/model/{model_name}")

    pred = np.zeros_like(y_valid)
    for cfips in track(np.unique(group_valid)):
        _X_valid = X_valid[group_valid == cfips]

        model = load_pickle(str(model_dir / f"{cfips}.pkl"))

        pred[group_valid == cfips] = model.predict(_X_valid)

    return pred


def evaluate(y_true, y_pred) -> float:
    score = smape(y_true, y_pred)
    return score


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    model_name = cfg.exp

    features = cfg.rf_model.features
    if cfg.exp == "xgb":
        features = cfg.rf_model.features

    feat = load_feature("./data/feature", features)
    feat = np.nan_to_num(feat, nan=0.0)

    group = load_pickle("./data/feature/cfips.pkl")
    dt = load_pickle("./data/feature/first_day_of_month.pkl").ravel()
    target = load_pickle("./data/feature/target.pkl")

    if cfg.is_eval:
        valid_start_month = "2022-11-01"
        valid_end_month = "2023-02-01"
    else:
        valid_start_month = "2023-02-01"
        valid_end_month = "2023-07-01"

    X_train = feat[dt < valid_start_month, :]
    X_valid = feat[(dt >= valid_start_month) & (dt < valid_end_month), :]
    y_train = target[dt < valid_start_month, :].ravel()
    y_valid = target[(dt >= valid_start_month) & (dt < valid_end_month), :].ravel()

    group_train = group[dt < valid_start_month, :].ravel()
    group_valid = group[(dt >= valid_start_month) & (dt < valid_end_month), :].ravel()

    pred_avg = np.zeros_like(y_valid)
    add_seed = [32, 42, 56, 1, 45]
    for _seed in add_seed:
        train(
            X_train,
            y_train,
            group_train,
            X_valid,
            y_valid,
            group_valid,
            model_name,
            seed=int(cfg.seed + _seed),
        )

        pred = predict(X_valid, y_valid, group_valid)
        pred = fill_last_mbd(pred, group_valid)
        pred_avg += pred / len(add_seed)

    if cfg.is_eval:
        score = evaluate(y_valid, pred_avg)
        print(score)

        with open(f"./data/model/{model_name}/cv_score.txt", "w") as file:
            file.write(str(round(score, 6)))


if __name__ == "__main__":
    with timer("Train"):
        main()
