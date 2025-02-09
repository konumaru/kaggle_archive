import pathlib
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from metric import mcrmse
from utils import timer
from utils.io import load_pickle


def get_oofs(cfg: DictConfig) -> Dict[str, np.ndarray]:
    oofs = {}
    for model_name in ["xgb", "lgbm"]:
        model_dir_suffix = f"{model_name}/seed={cfg.seed}/"
        model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix
        oof = load_pickle(model_dir / "oof.pkl")
        oofs[model_name] = oof
    return oofs


def get_tragets(cfg: DictConfig) -> np.ndarray:
    features_dir = pathlib.Path(cfg.path.feature)
    targets = []
    for target_name in ["content", "wording"]:
        target = load_pickle(features_dir / f"{target_name}.pkl")
        targets.append(target)
    return np.concatenate(targets, axis=1)


def calc_simple_average(oofs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(list(oofs.values()), axis=0)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    targets = get_tragets(cfg)
    oofs = get_oofs(cfg)
    oof = calc_simple_average(oofs)

    score = mcrmse(targets, oof)
    print(score)


if __name__ == "__main__":
    with timer("main.py"):
        main()
