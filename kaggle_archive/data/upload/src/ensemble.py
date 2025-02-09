import pathlib
from typing import Optional

import hydra
import joblib
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> Ridge:
    model = Ridge(
        alpha=20.0,
        fit_intercept=False,
        random_state=seed,
    )
    model.fit(X_train, y_train.to_numpy().flatten())
    print("Ridge Coef:", model.coef_)

    if save_filepath:
        joblib.dump(model, save_filepath + ".joblib")

    return model


def train(cfg: DictConfig) -> pl.DataFrame:
    model_name = "ridge"
    model_dir_suffix = f"{model_name}/seed={cfg.seed}/"
    save_dir = pathlib.Path(cfg.path.ensemble) / model_dir_suffix
    save_dir.mkdir(exist_ok=True, parents=True)

    X = load_oof(cfg.path.train)
    target: pl.DataFrame = load_pickle(
        f"{cfg.path.feature}/{cfg.target_name}.pkl"
    )
    fold: pl.DataFrame = load_pickle(f"{cfg.path.feature}/{cfg.fold_name}.pkl")

    oof = np.zeros(len(target))

    for i in range(cfg.n_splits):
        print(f"Fold {i}")

        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = X.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = X.filter(is_valid).to_pandas()

        model = fit_ridge(
            X_train,
            y_train,
            seed=cfg.seed,
            save_filepath=str(save_dir / str(i)),
        )
        oof[is_valid] = np.array(model.predict(X_valid)).flatten()

    save_pickle(str(save_dir / "oof.pkl"), oof)
    oof = X.with_columns(pl.Series(oof).alias("stack"))
    return oof


def load_oof(dirpath: str | pathlib.Path) -> pl.DataFrame:
    if isinstance(dirpath, str):
        dirpath = pathlib.Path(dirpath)

    oof = {}
    train_models = [
        # "xgb",
        "lgbm",
        "lgbm_weighted",
        "lgbm_drop",
        "cat",
        "cat_weighted",
        "cat_drop",
    ]
    seeds = [42]
    for model in train_models:
        for seed in seeds:
            oof[f"{model}_{seed=}"] = load_pickle(
                str(dirpath / model / f"{seed=}/oof.pkl")
            )
    return pl.DataFrame(oof)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    target = load_pickle(f"{cfg.path.feature}/{cfg.target_name}.pkl")
    # is_raw = load_pickle(f"{cfg.path.feature}/is_raw.pkl")

    oof = train(cfg)
    oof = oof.with_columns(pl.col("stack").clip(-1.0, 1.0))
    oof = oof.with_columns(
        pl.from_numpy(
            load_oof(cfg.path.train).mean_horizontal().to_numpy(), ["mean"]
        )
    )
    oof_np = oof.select("stack").to_numpy().flatten()
    save_pickle(str(pathlib.Path(cfg.path.ensemble) / "oof.pkl"), oof_np)

    save_dir = pathlib.Path(cfg.path.ensemble)
    save_dir.mkdir(exist_ok=True)

    # is_raw_flag = is_raw["is_raw"].eq(1).to_numpy()
    score = mean_squared_error(target.to_numpy(), oof_np, squared=False)
    print(f"\n\nRMSE: {score}")
    save_txt(
        str(save_dir / f"score_{score:.8f}.txt"),
        str(score),
    )


if __name__ == "__main__":
    with timer("ensemble.py"):
        main()
