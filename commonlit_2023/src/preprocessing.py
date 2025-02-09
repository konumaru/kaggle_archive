import pathlib

import gensim
import gensim.downloader
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold

from utils import timer


def download_gensim_model(cfg: DictConfig) -> None:
    output_dir = pathlib.Path(cfg.path.preprocessed)
    model_names = [
        "fasttext-wiki-news-subwords-300",
        "glove-wiki-gigaword-300",
    ]

    for model_name in model_names:
        gensim.downloader.load(model_name)

        vectors = gensim.downloader.load(model_name)
        vectors.save(str(output_dir / f"{model_name}"))  # type: ignore


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path(cfg.path.preprocessed)

    summaries_train = pd.read_csv("./data/raw/summaries_train.csv")
    prompts_train = pd.read_csv("./data/raw/prompts_train.csv")

    train = pd.merge(
        prompts_train, summaries_train, on="prompt_id", how="right"
    )
    cv = GroupKFold(n_splits=cfg.n_splits)

    train = train.assign(fold=0)
    for fold, (_, valid_index) in enumerate(
        cv.split(train, groups=train["prompt_id"])
    ):
        train.loc[valid_index, "fold"] = fold

    train.to_csv(output_dir / "train.csv", index=False)

    # download_gensim_model(cfg)


if __name__ == "__main__":
    with timer("main.py"):
        main()
