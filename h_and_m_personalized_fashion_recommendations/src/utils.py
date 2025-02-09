import os
import pickle
import random
import shutil
import time
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch


def feature(use_cache: bool = True):
    def _feature(func: Callable):
        def _wapper(*args, **kwargs):
            assert (
                "save_dir" in kwargs.keys()
            ), "Functions decorated with @feature must have save_dir as an argument"

            save_dir = kwargs["save_dir"]
            filepath = os.path.join(save_dir, func.__name__ + ".pkl")
            if use_cache == True and os.path.exists(filepath):
                with open(filepath, "rb") as file:
                    result = pickle.load(file)
            else:
                result = func(*args, **kwargs)
                with open(filepath, "wb") as file:
                    pickle.dump(result, file)

            return result

        return _wapper

    return _feature


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(filepath: str, data: Any):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def average_precision(target, predict, k=12):
    len_target = min(len(target), k)

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predict):
        if p in target and p not in predict[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len_target, k)


def mean_average_precision(targets, predicts, k=12):
    map_top_k = np.mean([average_precision(t, p) for t, p in zip(targets, predicts)])
    assert 0.0 <= map_top_k <= 1.0, "map_top_k must be 0.0 <= map_top_k <= 1.0"
    return map_top_k


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def mk_empty_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
