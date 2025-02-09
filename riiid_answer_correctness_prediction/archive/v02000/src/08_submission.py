import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import xgboost as xgb
import lightgbm as lgb

from utils.common import load_pickle, dump_pickle
from func import UserState, get_feature_and_update_state


is_kaggle_notebook = (
    True if os.path.exists("../input/riiid-test-answer-prediction") else False
)
kaggl_dir = "../input/riiidexternaldata"


"""
# Extarnal files.
- ../data/01_preprocessing/task_container_feature.csv
- ../data/02_transform/content.pkl
- ../data/04_create_feature/user_state.pkl
- ../data/05_train/xgb_models.pkl
- ../data/05_train/lgbm_models.pkl
- ../data/07_stack/stack_models.pkl
"""


def strList_to_intList(x, split_str=","):
    if x is np.nan or x == "[]":
        return list()
    else:
        x = x.split(split_str)
        x = [int(_x) for _x in x]
        return x


def create_features(data: pd.DataFrame):
    # External data.
    if is_kaggle_notebook:
        content = load_pickle(f"{kaggl_dir}/content.pkl")
        task_container = pd.read_csv(f"{kaggl_dir}/task_container_feature.csv")
        us_filepath = f"{kaggl_dir}/user_state.pkl"
    else:
        content = pd.read_pickle("../data/02_transform/content.pkl")
        task_container = pd.read_csv(
            "../data/01_preprocessing/task_container_feature.csv"
        )
        us_filepath = "../data/04_create_feature/user_state.pkl"
    # Define UserState class.
    us = UserState(is_test=True)
    us.load_state(us_filepath)

    data["answered_correctly"] = 0
    columns = ["prior_group_answers_correct", "prior_group_responses"]
    for c in columns:
        data[c] = data[c].str.extract("\[(.+)\]")
        data[c] = data[c].map(strList_to_intList)
    data["prior_question_had_explanation"] = (
        data["prior_question_had_explanation"]
        .map({True: 1, False: 0})
        .fillna(-999)
        .astype(int)
    )

    result = Parallel(n_jobs=-1)(
        delayed(get_feature_and_update_state)(us, idx, row)
        for idx, row in data.iterrows()
    )

    result = pd.DataFrame(result)
    result = result.loc[result["content_type_id"] == 0].reset_index(drop=True)
    X = result.drop(
        ["user_id", "row_id", "timestamp", "content_type_id", "answered_correctly"],
        axis=1,
    )

    X = X.merge(content, how="left", on="content_id")
    X = X.merge(task_container, how="left", on="task_container_id")
    X.fillna(-999, inplace=True)
    return X


def predict_xgb(models, data):
    pred = np.mean(
        [m.predict(xgb.DMatrix(data), ntree_limit=m.best_ntree_limit) for m in models],
        axis=0,
    )
    return pred


def predict_lgbm(models, data):
    pred = np.mean(
        [m.predict(data, num_iteration=m.best_iteration) for m in models], axis=0
    )
    return pred


def predict(data, xgb_models, lgb_models, stack_models):
    pred = np.array(
        [
            predict_xgb(xgb_models, data),
            predict_lgbm(lgb_models, data),
        ]
    ).T
    pred = pd.DataFrame(pred, columns=["xgb", "lgb"])
    pred = predict_xgb(stack_models, pred)
    return pred


def main():
    if is_kaggle_notebook:
        xgb_models = load_pickle(f"{kaggl_dir}/xgb_models.pkl")
        lgb_models = load_pickle(f"{kaggl_dir}/lgbm_models.pkl")
        stack_models = load_pickle(f"{kaggl_dir}/stack_models.pkl")
    else:
        xgb_models = load_pickle("../data/05_train/xgb_models.pkl")
        lgb_models = load_pickle("../data/05_train/lgbm_models.pkl")
        stack_models = load_pickle("../data/07_stack/stack_models.pkl")

    df = pd.read_csv("../data/raw/example_test.csv")
    for i, gdf in df.groupby("group_num"):
        # Initialize time-series-api data.
        test = gdf.set_index("group_num").copy()
        sample_prediction_df = test.loc[test["content_type_id"] == 0, ["row_id"]].copy()
        sample_prediction_df["answered_correctly"] = 0.0
        # Preprocessing test data.
        X_test = create_features(test)

        pred = predict(X_test, xgb_models, lgb_models, stack_models)

        submit = sample_prediction_df.copy(deep=True)
        submit["answered_correctly"] = pred
        # env.predict(submit.loc[submit["content_type_id"] == 0, :])

        print(submit.head())


if __name__ == "__main__":
    main()
