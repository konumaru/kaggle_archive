import os
import csv
import time
import random
import pickle
import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    druration_time = str(datetime.timedelta(seconds=time.time() - start_time))[:7]
    print(f"[{name}] done in {druration_time}\n")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_pickle(filepath, verbose: bool = True):
    if verbose:
        print(f"Load pickle from {filepath}.")
    with open(filepath, "rb") as file:
        return pickle.load(file)


def dump_pickle(data, filepath, verbose: bool = True):
    if verbose:
        print(f"Dump pickle to {filepath}.")
    with open(filepath, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_text(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    return text


def cache_result(filepath, use_cache=True):
    """Save result decorator.

    Parameters
    ----------
    filename : str
        filename, when save with pickle.
    use_cache : bool, optional
        Is use already cash result then pass method process, by default True
    """

    def _acept_func(func):
        def run_func(*args, **kwargs):
            if use_cache and os.path.exists(filepath):
                print(f"Load Cached data, {filepath}")
                return load_pickle(filepath)
            result = func(*args, **kwargs)

            print(f"Cache to {filepath}")
            dump_pickle(result, filepath)
            return result

        return run_func

    return _acept_func


def append_list_as_row(filename: str, list_of_elem: list):
    with open(filename, "a+", newline="") as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)


def get_logger(filename="log", dirpath="logs", mode="a"):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    # For print console.
    stream_hundler = StreamHandler()
    stream_hundler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(stream_hundler)
    # For dump log file.
    file_handler = FileHandler(filename=f"{dirpath}/{filename}.log", mode=mode)
    file_handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(file_handler)
    return logger


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                    and c_prec == np.finfo(np.float32).precision
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
