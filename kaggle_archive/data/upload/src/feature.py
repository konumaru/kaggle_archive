import itertools
import os
import pathlib

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold

from utils import timer
from utils.feature import BaseFeature, cache
from utils.io import load_pickle, load_txt

FEATURE_DIR = pathlib.Path("data/feature")
INPUT_DIR = pathlib.Path("data/preprocessing")


# =====================
# Target
# =====================


@cache(FEATURE_DIR)
def utility_agent1() -> pl.DataFrame:
    train = pl.read_parquet(INPUT_DIR / "train.parquet")
    target_col_name = "utility_agent1"
    target = train.select(target_col_name)
    return target


# =====================
# Fold
# =====================


@cache(FEATURE_DIR)
def fold(n_splits: int) -> pl.DataFrame:
    train = pl.read_parquet(INPUT_DIR / "train.parquet").to_pandas()
    col_group = "GameRulesetName"

    result = np.zeros(len(train))
    cv = GroupKFold(n_splits=n_splits)
    for i, (_, valid_idx) in enumerate(
        cv.split(train, None, train[col_group])
    ):
        result[valid_idx] = i
    return pl.DataFrame({"fold": pl.Series(result)})


# =====================
# Weight
# =====================


@cache(FEATURE_DIR)
def weight_inversed() -> pl.DataFrame:
    train = pl.read_parquet(INPUT_DIR / "train.parquet")
    d = {}
    for row in (
        train["utility_agent1"]
        .round(2)
        .value_counts()
        .sort(by="utility_agent1")
        .iter_rows()
    ):
        d[row[0]] = row[1] / len(train) * 10

    return train.select(
        pl.col("utility_agent1")
        .round(1)
        .replace_strict(d)
        .alias("weight_inversed")
    )


# =====================
# Is Raw
# =====================


@cache(FEATURE_DIR)
def is_raw() -> pl.DataFrame:
    train = pl.read_parquet(INPUT_DIR / "train.parquet")
    return train.select(pl.col("is_raw").cast(pl.Int32))


# =====================
# Feature
# =====================


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

    @cache(FEATURE_DIR)
    def numeric_feature(self) -> pl.DataFrame:
        data = self.data.clone()
        num_cols = [c for c in self.num_cols if c not in self.one_unique_cols]
        return data.select(sorted(num_cols)).cast(pl.Float32)

    @cache(FEATURE_DIR)
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

    @cache(FEATURE_DIR)
    def te_agents(self) -> pl.DataFrame:
        cols = [
            "agent1_expansion",
            "agent1_playout",
            "agent1_score_bounds",
            "agent1_selection",
            "agent2_expansion",
            "agent2_playout",
            "agent2_score_bounds",
            "agent2_selection",
        ]

        cols_group = []
        # cols_group += cols  # NOTE: Not work for me, cv score down 0.02
        cols_group += list(itertools.combinations(cols, 2))  # NOTE: -0.01
        # cols_group += list(itertools.combinations(cols, 3))  # NOTE: -0.015
        # cols_group += list(itertools.combinations(cols, 4))  # NOTE: -0.015

        data = self.data.clone()
        fold = load_pickle(FEATURE_DIR / "fold.pkl")
        data = data.with_columns(fold)
        data = data.with_columns(self.agent_parsed_feature())

        cols_select = []
        for col_group in cols_group:
            if isinstance(col_group, str):
                col_group = [col_group]
            elif isinstance(col_group, tuple):
                col_group = list(col_group)

            data = data.with_columns(
                pl.when(pl.col("utility_agent1") == 1.0)
                .then(1)
                .otherwise(0)
                .mean()
                .over(["fold"] + col_group)
                .alias(
                    "{}_te_one_mean".format("_".join(col_group)),
                ),
                pl.when(pl.col("utility_agent1") == 0.0)
                .then(1)
                .otherwise(0)
                .mean()
                .over(["fold"] + col_group)
                .alias(
                    "{}_te_zero_mean".format("_".join(col_group)),
                ),
            )
            cols_select.append("{}_te_one_mean".format("_".join(col_group)))
            cols_select.append("{}_te_zero_mean".format("_".join(col_group)))

        return data.select(cols_select)

    @cache(FEATURE_DIR)
    def is_raw(self) -> pl.DataFrame:
        return self.data.select(pl.col("is_raw").cast(pl.Float32))

    # @cache(FEATURE_DIR)
    # def text_embeddings(self) -> pl.DataFrame:
    #     vectorizer = load_pickle(self.input_dir / "tfidf_vectorizer.pkl")

    #     cols_text = ["EnglishRules", "LudRules"]
    #     embedding = vectorizer.transform(
    #         self.data.select(
    #             pl.concat_str(cols_text, separator=" ").alias("text")
    #         ).to_numpy()[:, 0]
    #     ).toarray()
    #     return pl.from_numpy(
    #         embedding, schema=[f"tfidf_{i}" for i in range(embedding.shape[1])]
    #     )

    # @cache(FEATURE_DIR)
    # def parsed_lud_rules(self) -> pl.DataFrame:
    #     data = self.data.clone()
    #     equiment = data.select(
    #         pl.col("LudRules")
    #         .str.extract(r"equipment \{ \((\w+) ", 1)
    #         .replace_strict(
    #             {
    #                 "board": 0,
    #                 "piece": 1,
    #                 "surakartaBoard": 2,
    #                 "boardless": 3,
    #                 "mancalaBoard": 4,
    #             },
    #             default=-1,
    #         )
    #         .alias("equipment")
    #     )
    #     return equiment


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    utility_agent1()
    weight_inversed()
    fold(n_splits=cfg.n_splits)
    # is_raw()

    input_dirpath = pathlib.Path(cfg.path.preprocessing)
    train = pl.read_parquet(input_dirpath / "train.parquet")

    one_unique_cols = load_txt(input_dirpath / "one_unique_cols.txt").split(
        "\n"
    )
    train = train.drop(one_unique_cols)

    feature = UMFeature(train, INPUT_DIR)
    feature.create_feature()


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
