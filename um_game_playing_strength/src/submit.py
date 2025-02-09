import os
import pathlib
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from utils import timer
from utils.feature import BaseFeature
from utils.io import load_txt

INPUT_DIR = pathlib.Path("data/upload")
FEATURE_DIR = INPUT_DIR
USE_CACHE = False
SAVE_CACHE = False


class UMFeature(BaseFeature):
    def __init__(
        self,
        data: pl.DataFrame,
        input_dir: str | pathlib.Path,
    ) -> None:
        self.data = data

        self.input_dir = pathlib.Path(input_dir)
        self.num_cols = load_txt(self.input_dir / "numeric_cols.txt").split(
            "\n"
        )
        self.one_unique_cols = load_txt(
            self.input_dir / "one_unique_cols.txt"
        ).split("\n")

    def numeric_feature(self) -> pl.DataFrame:
        data = self.data.clone()
        num_cols = [c for c in self.num_cols if c not in self.one_unique_cols]
        return data.select(sorted(num_cols)).cast(pl.Float32)

    def agent_parsed_feature(self) -> pl.DataFrame:
        expansion_keys = {"0.1": 0, "0.6": 1, "1.41421356237": 2}
        selectio_keys = {
            "ProgressiveHistory": 0,
            "UCB1": 1,
            "UCB1GRAVE": 2,
            "UCB1Tuned": 3,
        }
        playout_keys = {"Random200": 0, "MAST": 1, "NST": 2}

        _agent_parsed = [
            self.data[agent]
            .str.split_exact("-", 5)
            .struct.rename_fields(
                [
                    "mcts",
                    "selection",
                    "expansion",
                    "playout",
                    "score_bounds",
                ]
            )
            .alias(f"{agent}_parsed")
            .to_frame()
            .unnest(f"{agent}_parsed")
            .drop("mcts")
            .with_columns(
                pl.col("expansion").replace_strict(expansion_keys, default=-1)
            )
            .with_columns(
                pl.col("score_bounds").replace_strict(
                    {"true": 1, "false": 0}, default=-1
                )
            )
            .with_columns(
                pl.col("playout").replace_strict(playout_keys, default=-1)
            )
            .with_columns(
                pl.col("selection").replace_strict(selectio_keys, default=-1)
            )
            .rename(
                {
                    s: f"{agent}_{s}"
                    for s in [
                        "selection",
                        "expansion",
                        "playout",
                        "score_bounds",
                    ]
                }
            )
            for agent in ["agent1", "agent2"]
        ]

        agent_parsed = pl.concat(_agent_parsed, how="horizontal")
        return agent_parsed.cast(pl.Int32)

    def is_raw_feature(self) -> pl.DataFrame:
        self.data = self.data.with_columns(pl.lit(1).alias("is_raw"))
        return self.data.select(["is_raw"]).cast(pl.Float32)


class Predictor:
    def __init__(self, input_dir: str | pathlib.Path) -> None:
        self.input_dir = pathlib.Path(input_dir)
        self.drop_cols = load_txt(self.input_dir / "drop_cols.txt").split("\n")

        self.models_lgbm = self.load_lgbm_models()
        self.models_lgbm_weighted = self.load_lgbm_models("weighted")
        self.models_lgbm_drop = self.load_lgbm_models("drop")
        self.models_cat = self.load_cat_models()
        self.models_cat_weighted = self.load_cat_models("weighted")
        self.models_cat_drop = self.load_cat_models("drop")
        self.models_ridge = self.load_ridge_models()

        self.um_feature = UMFeature(pl.DataFrame(), input_dir)

    def predict(
        self, test: pl.DataFrame, sample_sub: pl.DataFrame
    ) -> pl.DataFrame:
        self.um_feature.data = test
        features: pl.DataFrame = self.um_feature.create_feature(
            return_type="polars"
        )  # type: ignore

        preds_lgbm = np.mean(
            [
                model.predict(
                    features.to_pandas(), num_iteration=model.best_iteration_
                )
                for model in self.models_lgbm
            ],
            axis=0,
        )
        preds_lgbm_weighted = np.mean(
            [
                model.predict(
                    features.to_pandas(), num_iteration=model.best_iteration_
                )
                for model in self.models_lgbm_weighted
            ],
            axis=0,
        )
        preds_lgbm_drop = np.mean(
            [
                model.predict(
                    features.drop(self.drop_cols).to_pandas(),
                    num_iteration=model.best_iteration_,
                )
                for model in self.models_lgbm_drop
            ],
            axis=0,
        )
        preds_cat = np.mean(
            [model.predict(features.to_pandas()) for model in self.models_cat],
            axis=0,
        )
        preds_cat_weighted = np.mean(
            [
                model.predict(features.to_pandas())
                for model in self.models_cat_weighted
            ],
            axis=0,
        )
        preds_cat_drop = np.mean(
            [
                model.predict(
                    features.drop(self.drop_cols).to_pandas(),
                )
                for model in self.models_cat_drop
            ],
            axis=0,
        )

        preds_first = pd.DataFrame(
            {
                "lgbm_seed=42": preds_lgbm,
                "lgbm_weighted_seed=42": preds_lgbm_weighted,
                "lgbm_drop_seed=42": preds_lgbm_drop,
                "cat_seed=42": preds_cat,
                "cat_weighted_seed=42": preds_cat_weighted,
                "cat_drop_seed=42": preds_cat_drop,
            }
        )

        preds = np.mean(
            [model.predict(preds_first) for model in self.models_ridge], axis=0
        )
        preds_clipped = np.clip(preds, -1.0, 1.0)
        return sample_sub.with_columns(
            pl.Series(preds_clipped).alias("utility_agent1")
        )

    def load_xgb_models(self, suffix: Optional[str] = None) -> List[Any]:
        if suffix:
            model_dir = pathlib.Path(self.input_dir / f"xgb_{suffix}/seed=42")
        else:
            model_dir = pathlib.Path(self.input_dir / "xgb/seed=42")
        models = []
        for i in range(5):
            model = XGBRegressor()
            model.load_model(str(model_dir / f"{i}.json"))
            models.append(model)
        return models

    def load_lgbm_models(self, suffix: Optional[str] = None) -> List[Any]:
        if suffix:
            model_dir = pathlib.Path(self.input_dir / f"lgbm_{suffix}/seed=42")
        else:
            model_dir = pathlib.Path(self.input_dir / "lgbm/seed=42")
        models = []
        for i in range(5):
            model = joblib.load(model_dir / f"{i}.joblib")
            models.append(model)
        return models

    def load_cat_models(self, suffix: Optional[str] = None) -> List[Any]:
        if suffix:
            model_dir = pathlib.Path(self.input_dir / f"cat_{suffix}/seed=42")
        else:
            model_dir = pathlib.Path(self.input_dir / "cat/seed=42")
        models = []
        for i in range(5):
            model = CatBoostRegressor()
            model.load_model(str(model_dir / f"{i}.cbm"))
            models.append(model)
        return models

    def load_ridge_models(self) -> List[Any]:
        model_dir = pathlib.Path(self.input_dir / "ridge/seed=42")
        models = []
        for i in range(5):
            model = joblib.load(model_dir / f"{i}.joblib")
            models.append(model)
        return models


def main() -> None:
    test: pl.DataFrame = pl.read_csv("data/raw/test.csv")
    smpl_sub = pl.read_csv("data/raw/sample_submission.csv")

    predictor = Predictor(INPUT_DIR)
    pred = predictor.predict(test, smpl_sub)

    assert (
        pred.shape == smpl_sub.shape
    ), f"Expected shape {smpl_sub.shape}, got {pred.shape}"

    print(pred)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
