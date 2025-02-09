import pathlib
from typing import Dict, List

import hydra
import polars as pl
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import timer
from utils.io import save_txt


def get_low_unique_cols(data: pl.DataFrame, threshold: int = 1) -> list[str]:
    cols = data.columns
    cols_drop = [
        "Id",
        "GameRulesetName",
        "agent1",
        "agent2",
        "EnglishRules",
        "LudRules",
        "num_wins_agent1",
        "num_draws_agent1",
        "num_losses_agent1",
        "utility_agent1",
    ]
    cols = list(set(cols) - set(cols_drop))

    cols += [
        col
        for col in data.columns
        if data.select(pl.col(col).null_count()).item() == data.height
    ]

    return [c for c in cols if data[c].n_unique() <= threshold]


def get_numeric_cols(data: pl.DataFrame) -> list[str]:
    cols = data.columns
    cols_str = [
        "Id",
        "GameRulesetName",
        "agent1",
        "agent2",
        "EnglishRules",
        "LudRules",
        "num_wins_agent1",
        "num_draws_agent1",
        "num_losses_agent1",
        "utility_agent1",
        "is_raw",
    ]

    cols = list(set(cols) - set(cols_str))
    return cols


def get_encoding_mapping(data: pl.DataFrame, col: str) -> Dict[str, int]:
    return dict(
        zip(sorted(data[col].unique().to_list()), range(data[col].n_unique()))
    )


def train_tfidf_enconder(
    data: pl.DataFrame, cols: List[str]
) -> TfidfVectorizer:
    X = data.select(
        pl.concat_str(cols, separator=" ").alias("text")
    ).to_numpy()
    vectorizer = TfidfVectorizer(max_features=512, ngram_range=(2, 3))
    vectorizer.fit(X[:, 0])
    return vectorizer


def concat_trains(
    train: pl.DataFrame, train_revsered: pl.DataFrame
) -> pl.DataFrame:
    train = train.with_columns(pl.lit(1).alias("is_raw"))
    train_revsered = (
        train_revsered.with_columns(pl.lit(0).alias("is_raw"))
        .select(train.columns)
        .with_columns((1 - pl.col("AdvantageP1")).alias("AdvantageP1"))
    )

    train_concat = pl.concat([train, train_revsered], how="vertical")
    return train_concat


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    raw_dirpath = pathlib.Path(cfg.path.raw)
    output_dirpath = pathlib.Path(cfg.path.preprocessing)
    # Load and dump parquet
    train = pl.read_csv(raw_dirpath / "train.csv")
    ## Reverse Agent1 and Agent2 for data augmentation
    train_reversed = train.rename(
        {"agent1": "agent2", "agent2": "agent1"}
    ).with_columns(pl.col("utility_agent1") * -1)
    train_reversed.write_parquet(output_dirpath / "train_reversed.parquet")
    train = concat_trains(train, train_reversed)
    train.write_parquet(output_dirpath / "train.parquet")

    concepts = pl.read_csv(raw_dirpath / "concepts.csv")
    concepts.write_parquet(output_dirpath / "concepts.parquet")

    # Get low unique cols
    one_unique_cols = get_low_unique_cols(train, 1)
    save_txt(
        output_dirpath / "one_unique_cols.txt", "\n".join(one_unique_cols)
    )

    # low_unique_cols = get_low_unique_cols(train, 2, 2)
    # encode_mapping = {
    #     col: get_encoding_mapping(train, col) for col in low_unique_cols
    # }
    # save_pickle(output_dirpath / "low_unique_encode_map.pkl", encode_mapping)

    numeric_cols = get_numeric_cols(train)
    save_txt(output_dirpath / "numeric_cols.txt", "\n".join(numeric_cols))

    # vectorizer = train_tfidf_enconder(train, ["EnglishRules", "LudRules"])
    # save_pickle(output_dirpath / "tfidf_vectorizer.pkl", vectorizer)


if __name__ == "__main__":
    with timer("preprocessing.py"):
        main()
