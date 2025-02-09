import functools
import inspect
import os
import pathlib
import pickle
from types import MethodType
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd


def feature(save_dir: str, use_cache: bool = True) -> Callable:
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        def run_func(*args, **kwargs) -> Any:
            filepath = os.path.join(save_dir, func.__name__ + ".pkl")

            if use_cache and os.path.exists(filepath):
                print(f"Use cached data, {filepath}")
                with open(filepath, "rb") as file:
                    data = pickle.load(file)
                    return data

            # NOTE: Run if not use or exist cache.
            print("Run Function of", func.__name__)
            result = func(*args, **kwargs)

            assert result.ndim == 2, "Feature dim must be 2d."
            with open(filepath, "wb") as file:
                pickle.dump(result, file)

            return result

        return run_func

    return wrapper


def load_feature(
    dirpath: Union[str, pathlib.Path], feature_names: List[str]
) -> np.ndarray:
    if isinstance(dirpath, str):
        saved_dir = pathlib.Path(dirpath)
    else:
        saved_dir = dirpath

    feats = []
    for feature_name in feature_names:
        filepath = str(saved_dir / (feature_name + ".pkl"))
        with open(filepath, "rb") as file:
            feat = pickle.load(file)
        feats.append(feat)

    return np.concatenate(feats, axis=1)


class BaseFeature:
    feature_dir: pathlib.Path = pathlib.Path("./data/tmp")
    use_cache: bool = True
    save_cache: bool = True

    def __init__(
        self,
        data: pd.DataFrame,
        use_cache: bool = True,
        is_test: bool = False,
        feature_dirpath: Union[None, str] = None,
    ) -> None:
        self.data = data
        self.is_test = is_test

        BaseFeature.use_cache = False if is_test else use_cache
        BaseFeature.save_cache = False if is_test else True

        if feature_dirpath is not None:
            BaseFeature.feature_dir = pathlib.Path(feature_dirpath)
            BaseFeature.feature_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache(use_cache: bool = True) -> Callable:
        def wrapper(func: Callable) -> Callable:
            @functools.wraps(func)
            def run_func(*args, **kwargs) -> Any:
                save_dir = BaseFeature.feature_dir
                filepath = os.path.join(save_dir, func.__name__ + ".pkl")

                if (
                    use_cache
                    and os.path.exists(filepath)
                    and BaseFeature.use_cache
                ):
                    print(f"Use cached data, {filepath}")
                    with open(filepath, "rb") as file:
                        data = pickle.load(file)
                        return data

                print("Run Function of", func.__name__)
                result = func(*args, **kwargs)

                assert result.ndim == 2, "Feature dim must be 2d."

                if BaseFeature.save_cache:
                    with open(filepath, "wb") as file:
                        pickle.dump(result, file)

                return result

            return run_func

        return wrapper

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
                "create_features",
                "get_feature_methods",
                "get_feature_names",
                "load_feature",
            ]
        ]
        feature_methods.sort(key=lambda x: x.__name__)
        return feature_methods

    def get_feature_names(self) -> List[str]:
        feature_methods = self.get_feature_methods()
        feature_names = [func.__name__ for func in feature_methods]
        return feature_names

    def create_features(self) -> np.ndarray:
        feature_funcs = self.get_feature_methods()

        features = []
        for func in feature_funcs:
            features.append(func())
        return np.concatenate(features, axis=1)

    def load_feature(self) -> np.ndarray:
        feature_names = self.get_feature_names()
        feats = []
        for feature_name in feature_names:
            filepath = str(self.feature_dir / (feature_name + ".pkl"))
            with open(filepath, "rb") as file:
                feat = pickle.load(file)
            feats.append(feat)
        return np.concatenate(feats, axis=1)
