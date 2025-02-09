import gc
import glob
import os
import pathlib
from datetime import datetime
from multiprocessing.context import assert_spawning
from typing import Dict, List

import numpy as np
import pandas as pd
import talib
from joblib import Parallel, delayed
from talib import abstract
from tqdm import tqdm

from store import CryptoStore

data_dir = pathlib.Path("../data")


def basic_faeture(batch, asset_data):
    batch_size = batch.shape[0]
    cols = [
        "open_sub_close",
        "hilow",
    ]

    dst = np.zeros((len(cols), 100))
    dst[0] = asset_data["open"] - asset_data["close"]
    dst[1] = (asset_data["high"] + asset_data["low"]) / 2
    return cols, dst[:, -batch_size:].T


def upper_shadow(batch, asset_data):
    batch_size = batch.shape[0]
    data = asset_data["high"] - np.maximum(asset_data["close"], asset_data["open"])
    return data[-batch_size:]


def lower_shadow(batch, asset_data):
    batch_size = batch.shape[0]
    data = np.minimum(asset_data["close"], asset_data["open"]) - asset_data["low"]
    return data[-batch_size:]


def momentum_indicators(batch, asset_data, timeperiod=15):
    momentum_indicators_columns = [
        "adx_mm",
        "adxr_mm",
        "aroondown_mm",
        "aroonup_mm",
        "bop_mm",
        "mfi_mm",
        "cci_mm",
        "cmo_mm",
        "rsi_mm",
        "mdi_mm",
        "pdi_mm",
        "mdm_mm",
        "pdm_mm",
        "dx_mm",
        "roc_mm",
        "rocp_mm",
        "will_mm",
    ]

    batch_size = batch.shape[0]
    out = np.zeros((17, 100))
    out[0] = talib.ADX(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Average Directional Movement Index
    out[1] = talib.ADXR(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Average Directional Movement Index Rating
    out[2:4] = talib.AROON(asset_data["high"], asset_data["low"], timeperiod=14)
    out[4] = talib.BOP(
        asset_data["open"], asset_data["high"], asset_data["low"], asset_data["close"]
    )  # Balance Of Power
    out[5] = talib.MFI(
        asset_data["high"],
        asset_data["low"],
        asset_data["close"],
        asset_data["volume"],
        timeperiod=14,
    )  # Money Flow Index
    out[6] = talib.CCI(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Commodity Channel Index
    out[7] = talib.CMO(asset_data["close"], timeperiod=14)  # Chande Momentum Oscillator
    out[8] = talib.RSI(asset_data["close"], timeperiod=14)  # Relative Strenght Index
    out[9] = talib.MINUS_DI(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Minus Directional Indicator
    out[10] = talib.PLUS_DI(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Plus Directional Indicator
    out[11] = talib.MINUS_DM(
        asset_data["high"], asset_data["low"], timeperiod=14
    )  # Minus Directional Movement
    out[12] = talib.PLUS_DM(
        asset_data["high"], asset_data["low"], timeperiod=14
    )  # Plus Directional Movement
    out[13] = talib.DX(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Directional Movement Index
    out[14] = talib.ROC(asset_data["close"], timeperiod=10)  # Rate of change
    out[15] = talib.ROCP(
        asset_data["close"], timeperiod=10
    )  # Rate of change Percentage
    out[16] = talib.WILLR(
        asset_data["high"], asset_data["low"], asset_data["close"], timeperiod=14
    )  # Williams' %R
    return momentum_indicators_columns, out[:, -batch_size:].T


def cycle_faeture(batch, asset_data):
    batch_size = batch.shape[0]
    cols = [
        "HT_DCPERIOD",
        "HT_DCPHASE",
        "HT_TRENDMODE",
    ]

    dst = np.zeros((len(cols), 100))
    dst[0] = talib.HT_DCPERIOD(asset_data["close"])
    dst[1] = talib.HT_DCPHASE(asset_data["close"])
    dst[2] = talib.HT_TRENDMODE(asset_data["close"])
    return cols, dst[:, -batch_size:].T


def row_plus_feature(batch, asset_data):
    def div(A, B):
        return np.divide(A, B, out=np.zeros_like(A), where=B != 0)

    batch_size = batch.shape[0]
    base = asset_data["count"]
    o = div(asset_data["open"], base)
    h = div(asset_data["high"], base)
    low = div(asset_data["low"], base)
    c = div(asset_data["close"], base)

    cols = [
        "base",
        "O",
        "H",
        "L",
        "C",
        "lowog_ret",
        "rs_volow",
        "trade",
        "gtrade",
    ]
    dst = np.zeros((len(cols), 100))

    dst[0] = base
    dst[1] = o
    dst[2] = h
    dst[3] = low
    dst[4] = c
    dst[5] = np.log1p(div(base, o))
    dst[6] = np.log1p(h) * np.log1p(div(h, o)) + np.log1p(div(low, c)) * np.log1p(
        div(low, o)
    )
    dst[7] = c - h
    dst[8] = div(dst[7], base)

    return cols, dst[:, -batch_size:].T


def make_feature(df: pd.DataFrame, store: CryptoStore):
    feature = pd.DataFrame(index=df.index)

    cols = ["Open", "High", "Low", "Close", "Volume", "VWAP"]
    df[cols] = df[cols].astype(np.float32)

    for i in range(14):
        is_asset_row = df["Asset_ID"] == i
        asset_data = store.get_asset_data(i)

        if is_asset_row.sum() > 0:
            feature.loc[is_asset_row, "Asset_ID"] = i

            cols, out = basic_faeture(df[is_asset_row], asset_data)
            feature.loc[is_asset_row, cols] = out
            feature.loc[is_asset_row, "UpperShadow"] = upper_shadow(
                df[is_asset_row], asset_data
            )
            feature.loc[is_asset_row, "LowerShadow"] = lower_shadow(
                df[is_asset_row], asset_data
            )

            cols, out = momentum_indicators(df[is_asset_row], asset_data, timeperiod=15)
            cols = [f"{c}_15" for c in cols]
            feature.loc[is_asset_row, cols] = out

            # cols, out = momentum_indicators(df[is_asset_row], asset_data, timeperiod=30)
            # cols = [f"{c}_30" for c in cols]
            # feature.loc[is_asset_row, cols] = out

            # cols, out = momentum_indicators(df[is_asset_row], asset_data, timeperiod=60)
            # cols = [f"{c}_60" for c in cols]
            # feature.loc[is_asset_row, cols] = out

            cols, out = cycle_faeture(df[is_asset_row], asset_data)
            feature.loc[is_asset_row, cols] = out

            cols, out = row_plus_feature(df[is_asset_row], asset_data)
            feature.loc[is_asset_row, cols] = out

    feature = feature.astype(np.float32)
    feature["Asset_ID"] = feature["Asset_ID"].astype(np.int8)
    return feature


def parallel_make_feature(
    batch: pd.DataFrame, store: CryptoStore
) -> List[pd.DataFrame]:
    store.update(batch)
    feature = make_feature(batch, store)
    return feature


def main():
    os.makedirs(data_dir / "feature", exist_ok=True)

    files = sorted(glob.glob(str(data_dir / "split/fold[0-9][0-9]_?????.pkl")))
    files.append("../data/split/test.pkl")
    for file in files:
        print("\nProcessing", file)
        data = pd.read_pickle(file)

        # NOTE: Like data processing of time series api.
        # store = CryptoStore()
        # for _, batch_df in tqdm(data.groupby("timestamp")):
        #     store.update(batch_df)
        #     features = make_feature(batch_df, store)
        #     # sample_submission["Target"] = predict(feature)

        store = CryptoStore()
        feature = Parallel(n_jobs=4)(
            delayed(parallel_make_feature)(batch, store)
            for _, batch in tqdm(data.groupby("timestamp"))
        )
        concat_feature = pd.concat(feature, axis=0, ignore_index=True)
        print(concat_feature.tail())
        print(concat_feature.shape)
        print(concat_feature.info())

        save_path = file.replace("split", "feature")
        concat_feature.to_pickle(save_path)

        del feature, concat_feature, store, data
        gc.collect()


if __name__ == "__main__":
    main()
