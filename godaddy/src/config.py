from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

hydra_dir = "./data/.hydra/"
hydra_config = {
    "mode": "MULTIRUN",
    "job": {
        "config": {"override_dirname": {"exclude_keys": ["{exp}", "{seed}"]}},
    },
    "run": {"dir": hydra_dir + "${exp}/${hydra.job.override_dirname}/seed=${seed}"},
    "sweep": {"dir": hydra_dir + "${exp}/${hydra.job.override_dirname}/seed=${seed}"},
}


@dataclass
class RandomForest:
    name: str = "rf"
    n_estimators: int = 200
    criterion: str = "absolute_error"
    features: List[str] = field(
        default_factory=lambda: [
            "lag_mbd",
            "lag_active",
            "pct_change_mbd",
            "pct_change_active",
            # "idx_month",  # NOTE: not improved.
            # "idx_month_of_year",  # NOTE: not improved.
            # "diff_mbd",  # NOTE: not improved.
            # "diff_active",  # NOTE: not improved.
            # "r2_mbd",  # NOTE: not improved.
            # "r2_active",  # NOTE: not improved.
            # "slope_mbd",  # NOTE: not improved.
            "slope_active",
            "window_mbd_mean",
            # "window_mbd_std",  # NOTE: not improved.
            # "window_mbd_sum",  # NOTE: not improved.
            # "window_active_mean",  # NOTE: not improved.
            # "window_active_std",  # NOTE: not improved.
            # "window_active_sum",  # NOTE: not improved.
            # "ewm_mbd",  # NOTE: not improved.
            # "ewm_active",  # NOTE: not improved.
            # "dif_mbd",  # NOTE: not improved.
            "lag_census",
            "pct_change_census",
            "population_growth_rate",
        ]
    )


@dataclass
class Config:
    hydra: DictConfig = OmegaConf.create(hydra_config)
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    seed: int = 42
    exp: str = "rf"

    is_eval: bool = True
    rf_model: RandomForest = field(default_factory=RandomForest)


cs: ConfigStore = ConfigStore.instance()
cs.store("config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
