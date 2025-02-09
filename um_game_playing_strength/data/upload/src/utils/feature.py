import functools
import inspect
import pathlib
import pickle
from types import MethodType
from typing import Any, Callable, List, Literal, Union

import numpy as np
import pandas as pd
import polars as pl


def cache(
    save_dir: Union[str, pathlib.Path],
    use_cache: bool = True,
    save_cache: bool = True,
) -> Callable[[Callable], Callable]:
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            filepath = save_dir / f"{func.__name__}.pkl"

            if use_cache and filepath.exists():
                print(f"Use cached data, {filepath}")
                return pickle.loads(filepath.read_bytes())

            print("Run Function of", func.__name__)
            result = func(*args, **kwargs)

            result = func(*args, **kwargs)
            result_dim = len(result.shape)
            assert result_dim == 2, "Feature dim must be 2d."

            if save_cache:
                filepath.write_bytes(pickle.dumps(result))

            return result

        return wrapper

    return decorator


def load_feature(
    dirpath: str | pathlib.Path, feature_names: List[str]
) -> pl.DataFrame:
    saved_dir = pathlib.Path(dirpath)

    feats = []
    for feature_name in feature_names:
        filepath = str(saved_dir / (feature_name + ".pkl"))
        with open(filepath, "rb") as file:
            feat = pickle.load(file)
        feats.append(feat)

    return pl.concat(feats, how="horizontal")


class BaseFeature:
    def __init__(self) -> None:
        pass

    def get_feature_methods(self) -> List[MethodType]:
        all_methods = [
            m[1] for m in inspect.getmembers(self, predicate=inspect.ismethod)
        ]
        feature_methods = [
            func
            for func in all_methods
            if func.__name__
            not in [
                "__init__",
                "get_feature_methods",
                "get_feature_names",
                "create_feature",
                "load_feature",
            ]
        ]
        feature_methods.sort(key=lambda x: x.__name__)
        return feature_methods

    def get_feature_names(self) -> List[str]:
        feature_methods = self.get_feature_methods()
        feature_names = [func.__name__ for func in feature_methods]
        return feature_names

    def create_feature(
        self, return_type: Literal["numpy", "pandas", "polars"] = "numpy"
    ):
        features = []
        for func in self.get_feature_methods():
            features.append(func())

        if return_type == "numpy":
            return np.concatenate(features, axis=1)
        elif return_type == "pandas":
            return pd.concat(features, axis=1)
        elif return_type == "polars":
            return pl.concat(features, how="horizontal")
