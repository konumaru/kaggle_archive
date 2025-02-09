import pathlib

import hydra
import numpy as np
import pandas as pd
from rich.progress import track

from config import Config
from postprocessing import fill_last_mbd, mult_growth_rate
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    model_name = cfg.exp

    features = cfg.rf_model.features
    if cfg.exp == "xgb":
        features = cfg.rf_model.features

    feat = load_feature("./data/feature", features)
    feat[np.isnan(feat)] = 0.0

    group = load_pickle("./data/feature/cfips.pkl")
    dt = load_pickle("./data/feature/first_day_of_month.pkl").ravel()
    growth_rate = load_pickle("./data/feature/population_growth_rate.pkl").ravel()

    test_start_month = "2022-11-01"

    X_test = feat[dt >= test_start_month, :]
    group_test = group[dt >= test_start_month, :].ravel()
    dt_test = dt[dt >= test_start_month]

    pred_avg = np.zeros(len(X_test))
    add_seed = [32, 42, 56, 1, 45]
    for _seed in add_seed:
        seed = int(cfg.seed + _seed)
        pred = np.zeros(len(X_test))
        for cfips in track(np.unique(group_test)):
            _X_test = X_test[group_test == cfips]

            model = load_pickle(f"./data/model/{model_name}/seed={seed}/{cfips}.pkl")
            pred[group_test == cfips] = model.predict(_X_test)

        pred_avg += pred / len(add_seed)

    pred = fill_last_mbd(pred_avg, group_test)

    submit = pd.DataFrame(
        {
            "cfips": group_test,
            "first_day_of_month": dt_test,
            "microbusiness_density": pred,
        }
    )
    submit["row_id"] = (
        submit["cfips"].astype(str) + "_" + submit["first_day_of_month"].astype(str)
    )
    submit = submit[["row_id", "microbusiness_density"]]
    submit = mult_growth_rate(submit)
    submit.sort_values("row_id")
    submit.to_csv(f"./data/submit/{model_name}/submission.csv", index=False)
    print(submit.head())


if __name__ == "__main__":
    with timer("Submission"):
        main()
