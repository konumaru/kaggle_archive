import os
import pathlib
from typing import Optional

import hydra
import joblib
import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, load_txt, save_pickle, save_txt


def fit_xgb(
    params,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: Optional[pd.DataFrame],
    y_valid: Optional[pd.DataFrame],
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=100,
    )
    if save_filepath:
        model.save_model(save_filepath + ".json")
    return model


def fit_lgbm(
    params,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: Optional[pd.DataFrame],
    y_valid: Optional[pd.DataFrame],
    weight_train: Optional[np.ndarray] = None,
    weight_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lightgbm.log_evaluation(100)],
        sample_weight=weight_train,
        eval_sample_weight=[weight_train, weight_valid],
    )
    if save_filepath:
        joblib.dump(model, save_filepath + ".joblib")
    return model


def fit_cat(
    params,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: Optional[pd.DataFrame],
    y_valid: Optional[pd.DataFrame],
    weight_train: Optional[np.ndarray] = None,
    weight_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> CatBoostRegressor:
    model = CatBoostRegressor(**params)
    model.set_params(random_seed=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
        sample_weight=weight_train,
    )
    if save_filepath:
        model.save_model(save_filepath + ".cbm", format="cbm")
    return model


def train(cfg: DictConfig) -> None:
    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    save_dir = pathlib.Path(cfg.path.train) / model_dir_suffix
    save_dir.mkdir(exist_ok=True, parents=True)

    feature = load_feature(cfg.path.feature, sorted(cfg.feature_names))
    print(feature)
    print("Feature shape:", feature.shape)

    target: pl.DataFrame = load_pickle(
        f"{cfg.path.feature}/{cfg.target_name}.pkl"
    )
    fold: pl.DataFrame = load_pickle(f"{cfg.path.feature}/{cfg.fold_name}.pkl")
    weight = load_pickle(f"{cfg.path.feature}/weight_inversed.pkl")

    if "drop" in cfg.model.name:
        drop_cols = load_txt(f"{cfg.path.preprocessing}/drop_cols.txt").split(
            "\n"
        )
        feature = feature.drop(drop_cols)

    oof = np.zeros(len(target))

    for i in range(cfg.n_splits):
        print(f"Fold {i}")

        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = feature.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = feature.filter(is_valid).to_pandas()
        y_valid = target.filter(is_valid).to_pandas()

        if "weighted" in cfg.model.name:
            weight_train = weight.filter(~is_valid).to_numpy().ravel()
            weight_valid = weight.filter(is_valid).to_numpy().ravel()
        else:
            weight_train = None
            weight_valid = None

        if cfg.model.name == "xgb":
            model = fit_xgb(
                cfg.model.params,
                X_train,
                y_train,
                X_valid,
                y_valid,
                save_filepath=str(save_dir / str(i)),
                seed=cfg.seed,
            )
            oof[is_valid] = model.predict(X_valid)
        elif "lgbm" in cfg.model.name:
            model = fit_lgbm(
                cfg.model.params,
                X_train,
                y_train,
                X_valid,
                y_valid,
                weight_train=weight_train,
                weight_valid=weight_valid,
                save_filepath=str(save_dir / str(i)),
                seed=cfg.seed,
            )
            oof[is_valid] = model.predict(
                X_valid, num_iteration=model.best_iteration_
            )
        elif "cat" in cfg.model.name:
            model = fit_cat(
                cfg.model.params,
                X_train,
                y_train,
                X_valid,
                y_valid,
                weight_train=weight_train,
                weight_valid=weight_valid,
                save_filepath=str(save_dir / str(i)),
                seed=cfg.seed,
            )
            oof[is_valid] = model.predict(X_valid)

    save_pickle(str(save_dir / "oof.pkl"), oof)


def evaluate(cfg: DictConfig) -> None:
    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    save_dir = pathlib.Path(cfg.path.train) / model_dir_suffix

    oof = load_pickle(str(save_dir / "oof.pkl"))
    target = load_pickle(f"{cfg.path.feature}/{cfg.target_name}.pkl")

    print("")
    fold: pl.DataFrame = load_pickle(f"{cfg.path.feature}/{cfg.fold_name}.pkl")
    for i in range(fold.n_unique()):
        is_valid = fold["fold"].eq(i).alias("is_valid")
        _target = target.filter(is_valid).to_numpy()
        _oof = oof[is_valid]
        score = mean_squared_error(_target, _oof, squared=False)
        print(f"Fold {i}: {score}")

    # score = np.mean(scores)
    score = mean_squared_error(target, oof, squared=False)
    print(f"\n\nRMSE: {score}")
    save_txt(
        str(save_dir / f"score_{score:.8f}.txt"),
        str(score),
    )


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
