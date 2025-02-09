import pathlib
from typing import Union

import hydra
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from feature import CommonLitFeature
from metric import mcrmse
from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def fit_rf(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_filepath: str,
    seed: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(X_train, y_train)
    save_pickle(str(f"{save_filepath}.pkl"), model)
    return model


def fit_svm(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_filepath: str,
    seed: int = 42,
) -> SVR:
    model = make_pipeline(StandardScaler(), SVR(**params))
    model = SVR(**params)
    model.fit(X_train, y_train)
    save_pickle(str(f"{save_filepath}.pkl"), model)
    return model


def fit_xgb(
    params,
    save_filepath: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )
    model.save_model(save_filepath + ".json")
    return model


def fit_lgbm(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    save_filepath: str,
    seed: int = 42,
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lightgbm.log_evaluation(50), lightgbm.early_stopping(50)],
    )
    model.booster_.save_model(  # type: ignore
        save_filepath + ".txt",
        num_iteration=model.best_iteration_,
        importance_type="gain",
    )
    return model


def get_first_stage_oof(cfg: DictConfig) -> np.ndarray:
    external_output_dir = pathlib.Path(cfg.path.external)

    filepaths = [
        # "deberta-v3-base/oof.csv",
        "finetune-debertav3-training/oof.csv",
        # "finetune-roberta-training/oof.csv",
    ]

    preds = []
    for filepath in filepaths:
        oof = pd.read_csv(str(external_output_dir / filepath))

        pred = oof[["pred_content", "pred_wording"]].to_numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=1)


def train(cfg: DictConfig) -> None:
    cl_feature = CommonLitFeature(
        pd.DataFrame(),
        sentence_encoder=SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda:0"
        ),
        feature_dir=cfg.path.feature,
    )
    text_features = cl_feature.load_feature()

    first_oof = get_first_stage_oof(cfg)
    features = np.concatenate([text_features, first_oof], axis=1)
    print(features.shape)

    features_dir = pathlib.Path(cfg.path.feature)
    folds = load_pickle(features_dir / "fold.pkl").ravel()

    oof = np.zeros(shape=(len(features), 2))
    targets = oof.copy()

    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix
    model_dir.mkdir(exist_ok=True, parents=True)

    for i, target_name in enumerate(["content", "wording"]):
        print("Target:", target_name)
        target = load_pickle(features_dir / f"{target_name}.pkl").ravel()
        targets[:, i] = target

        for fold in range(cfg.n_splits):
            print(f"Fold: {fold}")
            X_train = features[folds != fold]
            y_train = target[folds != fold]
            X_valid = features[folds == fold]
            y_valid = target[folds == fold]

            saved_filename = f"target={target_name}_fold={fold}"

            if cfg.model.name == "rf":
                model = fit_rf(
                    cfg.model.params,
                    X_train,
                    y_train,
                    str(model_dir / saved_filename),
                )
                oof[folds == fold, i] = model.predict(X_valid)
            elif cfg.model.name == "xgb":
                model = fit_xgb(
                    cfg.model.params,
                    str(model_dir / saved_filename),
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                )
                oof[folds == fold, i] = model.predict(X_valid)
            elif cfg.model.name == "lgbm":
                model = fit_lgbm(
                    cfg.model.params,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    str(model_dir / saved_filename),
                )
                oof[folds == fold, i] = model.predict(X_valid)
            elif cfg.model.name == "svm":
                model = fit_svm(
                    cfg.model.params,
                    X_train,
                    y_train,
                    str(model_dir / saved_filename),
                )
                oof[folds == fold, i] = model.predict(X_valid)

    save_pickle(str(model_dir / "oof.pkl"), oof)


def evaluate(cfg: DictConfig) -> None:
    features_dir = pathlib.Path(cfg.path.feature)

    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix
    model_dir.mkdir(exist_ok=True, parents=True)

    oof = load_pickle(str(model_dir / "oof.pkl"))

    targets = np.zeros_like(oof)
    for i, target_name in enumerate(["content", "wording"]):
        target = load_pickle(features_dir / f"{target_name}.pkl").ravel()
        targets[:, i] = target

    score = mcrmse(targets, oof)
    print(score)

    save_txt(
        str(model_dir / "score.txt"),
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
    with timer("main.py"):
        main()
