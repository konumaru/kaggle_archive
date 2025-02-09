import numpy as np
import optuna
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from utils.feature import load_feature
from utils.io import load_pickle, save_pickle


def load_data(i: int = 0):
    feature = load_feature(
        "data/feature",
        sorted(["agent_parsed_feature", "numeric_feature"]),
    )
    print("Feature shape:", feature.shape)
    target: pl.DataFrame = load_pickle("data/feature/utility_agent1.pkl")
    fold: pl.DataFrame = load_pickle("data/feature/fold.pkl")

    is_valid = fold["fold"].eq(i).alias("is_valid")

    X_train = feature.filter(~is_valid).to_pandas()
    y_train = target.filter(~is_valid).to_pandas()
    X_valid = feature.filter(is_valid).to_pandas()
    y_valid = target.filter(is_valid).to_pandas()

    return X_train, y_train, X_valid, y_valid


def objective_cat(trial: optuna.Trial):
    params = {
        "task_type": "GPU",  # GPUが使えない場合はCPUに設定
        "loss_function": "RMSE",
        "iterations": trial.suggest_int("iterations", 1200, 2000),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.5, 1.0
        ),
        "max_depth": trial.suggest_int("max_depth", 9, 15),
        # "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "random_strength": trial.suggest_float("random_strength", 3, 8),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 5, 10),
        "od_type": "IncToDec",
        "random_seed": 42,
    }

    feature = load_feature(
        "data/feature",
        sorted(["agent_parsed_feature", "numeric_feature"]),
    )
    target: pl.DataFrame = load_pickle("data/feature/utility_agent1.pkl")
    fold: pl.DataFrame = load_pickle("data/feature/fold.pkl")

    oof = np.zeros(len(target))
    for i in range(5):
        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = feature.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = feature.filter(is_valid).to_pandas()

        model = CatBoostRegressor(**params, silent=True)
        model.fit(X_train, y_train)

        oof[is_valid] = model.predict(X_valid)

    rmse = mean_squared_error(target.to_numpy(), oof, squared=False)  # type: ignore

    return float(rmse)


def objective_lgbm(trial: optuna.Trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        "num_leaves": trial.suggest_int("num_leaves", 32, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0),
        "force_col_wise": True,
        "seed": 42,
    }

    feature = load_feature(
        "data/feature",
        sorted(["agent_parsed_feature", "numeric_feature"]),
    )
    target: pl.DataFrame = load_pickle("data/feature/utility_agent1.pkl")
    fold: pl.DataFrame = load_pickle("data/feature/fold.pkl")
    weight = load_pickle("data/feature/weight_inversed.pkl")

    oof = np.zeros(len(target))
    for i in range(5):
        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = feature.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = feature.filter(is_valid).to_pandas()
        y_valid = target.filter(is_valid).to_pandas()

        use_weight = True
        if use_weight:
            weight_train = weight.filter(~is_valid).to_numpy().ravel()
        else:
            weight_train = None

        model = LGBMRegressor(**params, silent=True, verbose=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            sample_weight=weight_train,
        )
        oof[is_valid] = model.predict(X_valid)

    rmse = mean_squared_error(target.to_numpy(), oof, squared=False)  # type: ignore

    return float(rmse)


def objective_xgb(trial: optuna.Trial):
    params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": 2000,
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-5, 100),
        "lambda": trial.suggest_float("lambda", 1e-5, 100),
        "seed": 42,
    }

    feature = load_feature(
        "data/feature",
        sorted(["agent_parsed_feature", "numeric_feature"]),
    )
    target: pl.DataFrame = load_pickle("data/feature/utility_agent1.pkl")
    fold: pl.DataFrame = load_pickle("data/feature/fold.pkl")
    weight = load_pickle("data/feature/weight_inversed.pkl")

    oof = np.zeros(len(target))
    for i in range(5):
        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = feature.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = feature.filter(is_valid).to_pandas()
        y_valid = target.filter(is_valid).to_pandas()

        use_weight = False
        if use_weight:
            weight_train = weight.filter(~is_valid).to_numpy().ravel()
        else:
            weight_train = None

        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=0,
            sample_weight=weight_train,
        )

        oof[is_valid] = model.predict(X_valid)

    rmse = mean_squared_error(target.to_numpy(), oof, squared=False)  # type: ignore

    return float(rmse)


study = optuna.create_study(direction="minimize")
study.optimize(objective_cat, n_trials=50)
print("Best hyperparameters:", study.best_params)
print("Best RMSE:", study.best_value)

save_pickle("data/catboost.pkl", study.best_params)
