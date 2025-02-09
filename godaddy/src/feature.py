import numpy as np
import pandas as pd
from scipy.stats import linregress

from utils.feature import feature

FEATURE_DIR = str("./data/feature/")


@feature(FEATURE_DIR)
def target(data: pd.DataFrame) -> np.ndarray:
    return data[["microbusiness_density"]].to_numpy()


@feature(FEATURE_DIR)
def cfips(data: pd.DataFrame) -> np.ndarray:
    return data[["cfips"]].to_numpy()


@feature(FEATURE_DIR)
def first_day_of_month(data: pd.DataFrame) -> np.ndarray:
    return data[["first_day_of_month"]].to_numpy()


@feature(FEATURE_DIR)
def lag_mbd(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        lag = data.groupby("cfips")["microbusiness_density"].shift(n_lag + 1)
        result[:, n_lag] = lag.bfill().to_numpy()
    return result


@feature(FEATURE_DIR)
def pct_change_mbd(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        lag = data.groupby("cfips")["microbusiness_density"].pct_change(
            n_lag + 1, fill_method="ffill"
        )
        result[:, n_lag] = lag.bfill().to_numpy()
    result = np.clip(result, -0.05, 0.05)
    return result


@feature(FEATURE_DIR)
def pct_change_active(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        lag = data.groupby("cfips")["active"].pct_change(n_lag + 1, fill_method="ffill")
        result[:, n_lag] = lag.bfill().to_numpy()

    result = np.clip(result, -0.05, 0.05)
    return result


@feature(FEATURE_DIR)
def lag_active(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        lag = data.groupby("cfips")["active"].shift(n_lag + 1)
        result[:, n_lag] = lag.bfill().to_numpy()
    return result


@feature(FEATURE_DIR)
def diff_mbd(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        diff = data.groupby("cfips")["microbusiness_density"].diff(n_lag + 1)  # type: ignore
        result[:, n_lag] = diff.bfill().to_numpy()
    return result


@feature(FEATURE_DIR)
def diff_active(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), num_lag))
    for n_lag in range(num_lag):
        diff = data.groupby("cfips")["active"].diff(n_lag + 1)  # type: ignore
        result[:, n_lag] = diff.bfill().to_numpy()
    return result


@feature(FEATURE_DIR)
def window_mbd_mean(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["microbusiness_density"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).mean()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def window_mbd_sum(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["microbusiness_density"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).sum()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def window_mbd_std(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["microbusiness_density"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).std()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def window_active_mean(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["active"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).mean()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def window_active_sum(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["active"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).sum()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def window_active_std(data: pd.DataFrame) -> np.ndarray:
    num_windows = 10
    result = np.zeros((len(data), num_windows))
    for win in range(num_windows):
        win_mean = data.groupby("cfips")["active"].transform(
            lambda s: s.rolling(win + 1, min_periods=1).std()
        )
        result[:, win] = win_mean.to_numpy()
    return result


@feature(FEATURE_DIR)
def slope_mbd(data: pd.DataFrame) -> np.ndarray:
    def _calc_slope(x):
        slope, _, _, _, _ = linregress(range(len(x)), x)
        return slope

    min_month = 9
    slope = data.groupby("cfips")["microbusiness_density"].transform(
        lambda s: s.rolling(12, min_periods=min_month).apply(_calc_slope)
    )
    return slope.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def r2_mbd(data: pd.DataFrame) -> np.ndarray:
    def _calc_r2(x):
        _, _, r_value, _, _ = linregress(range(len(x)), x)
        return r_value

    min_month = 9
    r_value = data.groupby("cfips")["microbusiness_density"].transform(
        lambda s: s.rolling(12, min_periods=min_month).apply(_calc_r2)
    )
    return r_value.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def slope_active(data: pd.DataFrame) -> np.ndarray:
    def _calc_slope(x):
        slope, _, _, _, _ = linregress(range(len(x)), x)
        return slope

    min_month = 9
    slope = data.groupby("cfips")["active"].transform(
        lambda s: s.rolling(12, min_periods=min_month).apply(_calc_slope)
    )
    return slope.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def r2_active(data: pd.DataFrame) -> np.ndarray:
    def _calc_r2(x):
        _, _, r_value, _, _ = linregress(range(len(x)), x)
        return r_value

    min_month = 9
    r_value = data.groupby("cfips")["active"].transform(
        lambda s: s.rolling(12, min_periods=min_month).apply(_calc_r2)
    )
    return r_value.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def ewm_mbd(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), 5))
    for i, n_lag in enumerate(range(2, num_lag + 1, 2)):
        ewm = data.groupby("cfips")["microbusiness_density"].transform(
            lambda x: x.shift(n_lag).bfill().ewm(alpha=0.85).mean()
        )
        result[:, i] = ewm.to_numpy()
    return result


@feature(FEATURE_DIR)
def ewm_active(data: pd.DataFrame) -> np.ndarray:
    num_lag = 10
    result = np.zeros((len(data), 5))
    for i, n_lag in enumerate(range(2, num_lag + 1, 2)):
        ewm = data.groupby("cfips")["active"].transform(
            lambda x: x.shift(n_lag).bfill().ewm(alpha=0.85).mean()
        )
        result[:, i] = ewm.to_numpy()
    return result


@feature(FEATURE_DIR)
def dif_mbd(data: pd.DataFrame) -> np.ndarray:
    lag = 1
    data[f"mbd_lag_{lag}"] = (
        data.groupby("cfips")["microbusiness_density"].shift(lag).bfill()
    )
    data["dif"] = (data["microbusiness_density"] / data[f"mbd_lag_{lag}"]).fillna(
        1
    ).clip(0, None) - 1
    data.loc[(data[f"mbd_lag_{lag}"] == 0), "dif"] = 0
    data.loc[
        (data[f"microbusiness_density"] > 0) & (data[f"mbd_lag_{lag}"] == 0), "dif"
    ] = 1
    result = data[["dif"]].abs().to_numpy()
    return result


@feature(FEATURE_DIR)
def idx_month(data: pd.DataFrame) -> np.ndarray:
    result = data.groupby("cfips")["first_day_of_month"].cumcount()
    return result.to_numpy().reshape(-1, 1)


@feature(FEATURE_DIR)
def idx_month_of_year(data: pd.DataFrame) -> np.ndarray:
    result = data.groupby("cfips")["first_day_of_month"].cumcount()
    result = result % 12
    return result.to_numpy().reshape(-1, 1)


# ==============================
# External
# ==============================


@feature(FEATURE_DIR)
def lag_census(data: pd.DataFrame) -> np.ndarray:
    census = pd.read_csv("./data/preprocessing/census.csv")
    census.sort_values(by=["cfips", "year"], inplace=True)
    census.reset_index(drop=True, inplace=True)
    census.reset_index(inplace=True)

    feat_cols = census.columns.tolist()
    feat_cols.remove("cfips")
    feat_cols.remove("year")

    lags = []
    num_lag = 2
    for n_lag in range(num_lag):
        lag = census.groupby("cfips")[feat_cols].shift(n_lag + 1).ffill()
        lags.append(lag.to_numpy())

    feat_lag = np.concatenate(lags, axis=1)

    tmp = data[["cfips", "year"]].merge(
        census[["cfips", "year", "index"]], how="left", on=["cfips", "year"]
    )

    return feat_lag[tmp["index"].to_numpy()]


@feature(FEATURE_DIR)
def pct_change_census(data: pd.DataFrame) -> np.ndarray:
    census = pd.read_csv("./data/preprocessing/census.csv")
    census.sort_values(by=["cfips", "year"], inplace=True)
    census.reset_index(drop=True, inplace=True)
    census.reset_index(inplace=True)

    feat_cols = census.columns.tolist()
    feat_cols.remove("cfips")
    feat_cols.remove("year")

    lags = []
    num_lag = 2
    for n_lag in range(num_lag):
        lag = census.groupby("cfips")[feat_cols].pct_change(
            n_lag + 1, fill_method="ffill"
        )
        lags.append(lag.to_numpy())

    feat_lag = np.concatenate(lags, axis=1)

    tmp = data[["cfips", "year"]].merge(
        census[["cfips", "year", "index"]], how="left", on=["cfips", "year"]
    )

    feat = feat_lag[tmp["index"].to_numpy()]
    feat = np.nan_to_num(feat, nan=0.0)
    feat = np.clip(feat, None, 500)
    return feat


@feature(FEATURE_DIR)
def population_growth_rate(data: pd.DataFrame) -> np.ndarray:
    census_ex = pd.read_csv("./data/preprocessing/census_ex.csv")
    data = data.merge(census_ex, how="left", on=["cfips", "year"])
    groth_rate = 1.0 - data.groupby("cfips")["S0101_C01_026E"].pct_change(1)
    return groth_rate.to_numpy().reshape(-1, 1)


def main() -> None:
    data = pd.read_csv("./data/preprocessing/data.csv")
    print(data.head())
    print("-" * 50, "\n")

    feat_funcs = [
        target,
        cfips,
        first_day_of_month,
        lag_mbd,
        lag_active,
        pct_change_mbd,
        pct_change_active,
        diff_mbd,
        diff_active,
        window_mbd_mean,
        window_mbd_sum,
        window_mbd_std,
        window_active_mean,
        window_active_sum,
        window_active_std,
        slope_mbd,
        r2_mbd,
        slope_active,
        r2_active,
        ewm_mbd,
        ewm_active,
        dif_mbd,
        idx_month,
        idx_month_of_year,
        lag_census,
        pct_change_census,
        population_growth_rate,
    ]

    for func in feat_funcs:
        func(data)


if __name__ == "__main__":
    main()
