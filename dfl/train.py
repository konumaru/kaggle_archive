import glob
import os
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection

from metric import event_detection_ap, tolerances


def time2frame(time: float, fps: int = 25) -> int:
    frame = int(time * fps)
    return frame


def frame2time(frame: int, fps: int = 25) -> float:
    time = frame / fps
    return time


def load_data():
    data = []
    video_base_dir = "./data/raw/train/"
    video_filepath_list = glob.glob(os.path.join(video_base_dir, "*.mp4"))

    for video_filepath in video_filepath_list:
        video_id = os.path.basename(video_filepath).split(".")[0]
        save_dir = f"./data/feature/{video_id}"
        _data = pd.read_pickle(os.path.join(save_dir, "features.pickle"))
        _data["video_id"] = video_id
        data.append(_data)

    data = pd.concat(data, axis=0)
    return data


def create_dataset(data, labels):
    data = data.merge(
        labels[["video_id", "frame", "event", "game_id"]],
        how="right",
        on=["video_id", "frame"],
    )
    data["event"].fillna("background", inplace=True)

    label_map = {
        "challenge": 0,
        "throwin": 1,
        "play": 2,
        "background": 3,
        "start": 3,
        "end": 3,
    }
    data = data.assign(event=data["event"].map(label_map))
    return data


def gen_random_dummy_labels(labels: pd.DataFrame) -> pd.DataFrame:
    remove_time = labels["time"].to_numpy()

    dst = []
    for video_id in labels["video_id"].unique():
        s_e_labels = labels.query("video_id==@video_id and event in ['start', 'end']")
        s_e_times_np = s_e_labels["time"].to_numpy()
        dummy_labels = [
            (
                video_id,
                random.uniform(s_e_times_np[i], s_e_times_np[i + 1]),
                "background",
                np.nan,
            )
            for i in range(0, len(s_e_times_np), 2)
            if np.random.rand() < 0.01
        ]
        s_e_times = s_e_labels.iloc[1:-1]["time"].to_numpy()
        dummy_labels += [
            (
                video_id,
                random.uniform(s_e_times[i], s_e_times[i + 1]),
                "background",
                np.nan,
            )
            for i in range(0, len(s_e_times), 2)
            if np.random.rand() < 0.01
        ]

        dummy_labels = pd.DataFrame(
            columns=["video_id", "time", "event", "event_attributes"], data=dummy_labels
        )
        dst.append(dummy_labels)
    dst = pd.concat(dst, axis=0)
    dst = dst[~dst["time"].isin(remove_time)]
    return dst


def get_weight_map(num_of_sample):
    ones = np.ones(len(num_of_sample))
    weight = ones - (num_of_sample / np.sum(num_of_sample))
    weight_map = {i: w for i, w in enumerate(weight)}
    return weight_map


def main():
    labels = pd.read_csv("./data/raw/train.csv")
    dummy_labels = gen_random_dummy_labels(labels)
    labels = pd.concat((labels, dummy_labels), axis=0)
    labels = labels.assign(
        frame=labels["time"].apply(time2frame),
        game_id=labels["video_id"].str.split("_").apply(lambda x: x[0]),
    )

    data = load_data()

    dataset = create_dataset(data, labels)

    feature_cols = dataset.drop(
        ["frame", "video_id", "event", "game_id"], axis=1
    ).columns.tolist()
    X = dataset[list(set(feature_cols))].to_numpy()
    y = dataset["event"]
    group = dataset["game_id"].to_numpy()

    print(X.shape)

    label_cnt = dataset.event.value_counts().sort_index().tolist()
    weight_map = get_weight_map(label_cnt)
    weight_map[0] = weight_map[0] * 5.0
    weight_map[1] = weight_map[1] * 7.0
    weight_map[2] = weight_map[2] * 2.0
    weight_map[3] = weight_map[3] * 0.2
    weight = y.map(weight_map)

    cv = model_selection.GroupKFold(n_splits=3)

    oof = np.zeros(shape=(len(y), 2))
    for n_fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, group)):
        X_train, y_train = X[train_idx], y.iloc[train_idx].to_numpy()
        X_valid, y_valid = X[valid_idx], y.iloc[valid_idx].to_numpy()
        w_train = weight.iloc[train_idx].to_numpy()

        # NOTE: document url is below.
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
        bst = xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=100,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=1.0,
            random_state=42,
            min_child_weight=10,
            verbosity=0,
            # early_stopping_rounds=50,
        )
        bst = bst.fit(
            X=X_train,
            y=y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=50,
        )
        bst.save_model(f"./data/model/xgb_{n_fold}fold.json")

        proba = bst.predict_proba(X_valid)
        oof[valid_idx, 0] = proba.argmax(axis=1)
        oof[valid_idx, 1] = proba.max(axis=1)

    reverse_label_map = {
        0: "challenge",
        1: "throwin",
        2: "play",
        3: "background",
    }

    predict = pd.DataFrame(oof, columns=["event", "score"])
    predict = predict.assign(
        video_id=dataset["video_id"],
        time=dataset["frame"].apply(frame2time),
        event=predict["event"].map(reverse_label_map),
    )

    predict = predict[["video_id", "time", "event", "score"]]
    predict = predict.query(
        """event in ['challenge', 'throwin', 'play']"""
    ).reset_index(drop=True)
    predict = predict.query(
        """
        event == 'play' \
            | (event == 'throwin' & score > 0.0) \
            | (event == 'challenge' & score > 0.0)
        """
    )
    predict.to_csv("./data/tmp/predict.csv", index=False)
    print(
        predict.query("event not in ['start', 'end']")["event"]
        .value_counts()
        .sort_index()
    )

    print("\n\n=== Evaluation ===")
    solution = pd.read_csv(
        "./data/raw/train.csv", usecols=["video_id", "time", "event"]
    )
    print(
        solution.query("event not in ['start', 'end']")["event"]
        .value_counts()
        .sort_index()
    )
    score = event_detection_ap(solution, predict, tolerances)
    print(score)


if __name__ == "__main__":
    main()
