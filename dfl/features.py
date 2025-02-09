import glob
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from yolov7.custom_detect import detect

warnings.filterwarnings("ignore")


def detect_objct_position(video_filepath):

    with torch.no_grad():
        dst = detect(
            source=video_filepath,
            weights="data/yolov7.pt",
            device="0",
            imgsz=1280,
        )

    data = pd.DataFrame(dst, columns=["frame", "obj_name", "x", "y", "conf"])
    obj_name_str = data["obj_name"].map({0: "person", 32: "ball"})
    data = data.assign(obj_name=obj_name_str)
    return data


def calc_distance(a: Tuple, b: Tuple):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def ball(obj_pos):
    ball_pos = obj_pos.query("obj_name=='ball'")
    ball_pos = ball_pos.loc[ball_pos.groupby("frame")["conf"].idxmax()]
    ball_pos = ball_pos.pivot_table(
        index=["frame"],
        columns=["obj_name"],
        values=["x", "y", "conf"],  # type: ignore
    ).reset_index()
    frame = pd.DataFrame({"frame": list(range(0, ball_pos["frame"].max()))})
    ball_pos = frame.merge(ball_pos, how="left", on="frame").reset_index(drop=True)
    ball_pos.columns = [c if "" in c else c[1] + "_" + c[0] for c in ball_pos.columns]
    # NOTE: Fill NaNs with centrally moving averaged values.
    flag_nan = ball_pos.isnull().any(axis=1)
    ball_pos[flag_nan] = ball_pos.rolling(3, center=True).mean().fillna(0)[flag_nan]
    ball_pos = ball_pos.assign(
        frame=ball_pos["frame"].astype("int"), ball_is_nan=flag_nan.astype("int")
    )
    return ball_pos


def person(obj_pos, ball_pos):
    """
    Person feature is x, y, conf, distance (from ball position).
    """

    person_pos = obj_pos.query("obj_name=='person'")[["frame", "x", "y", "conf"]]
    person_pos = person_pos.query("0.05 < x < 0.95 & 0.02 < y < 0.98 | conf > 0.6")
    person_pos = person_pos[person_pos["frame"].isin(ball_pos.index)]

    ball_pos = ball_pos.loc[person_pos["frame"], ["ball_x", "ball_y"]].to_numpy()

    person_pos["dist"] = calc_distance(
        (person_pos["x"], person_pos["y"]), (ball_pos[:, 0], ball_pos[:, 1])
    )
    person_pos["rank"] = person_pos.groupby("frame")["dist"].rank().astype(int)
    person_pos = person_pos.query("rank < 24")
    person_pos = person_pos.assign(obj_name="person_" + person_pos["rank"].astype(str))
    person_pos = person_pos.pivot_table(
        index=["frame"],
        columns=["obj_name"],
        values=["x", "y", "dist"],  # type: ignore
    ).reset_index()
    person_pos.columns = [c[0] if "" in c else "_".join(c) for c in person_pos.columns]

    # NOTE: Add stats features.
    for xy in ["x", "y", "dist"]:
        person_pos = person_pos.assign(
            **{
                f"all_person_{xy}_{func}": person_pos[
                    [f"{xy}_person_{i}" for i in range(1, 24)]
                ].agg(func, axis=1)
                for func in ["mean", "min", "max", "std", "median"]
            }
        )

    person_pos.drop(
        [f"{xy}_person_{i}" for i in range(1, 24) for xy in ["x", "y"]],
        axis=1,
        inplace=True,
    )
    return person_pos


def calc_movement(data, periods: List = [1]):
    target_cols = data.columns[
        data.columns.str.contains("x")
        | data.columns.str.contains("y")
        | data.columns.str.contains("conf")
        | data.columns.str.contains("dist")
    ]
    dst = []
    for p in periods:
        _dst = data[target_cols].diff(p)
        _dst.columns = [f"{c}_diff_p{p}" for c in _dst.columns]
        dst.append(_dst)

    dst = pd.concat(dst, axis=1)
    return dst


def calc_arccos(
    ball_pos: pd.DataFrame, ball_movement: pd.DataFrame, periods: List = [1]
):
    v1 = np.sqrt(
        ((ball_pos[["ball_x", "ball_y"]] - np.array([0.5, 0.5])) ** 2)
        .sum(axis=1)
        .to_numpy()
    )

    dst = {}
    for p in periods:
        v2 = np.sqrt(
            (
                (
                    ball_movement[[f"ball_x_diff_p{p}", f"ball_y_diff_p{p}"]]
                    - np.array([0.5, 0.5])
                )
                ** 2
            )
            .sum(axis=1)
            .to_numpy()
        )
        dst[f"ball_prod_p{p}"] = np.arccos(np.divide(v1, v2))
    dst = pd.DataFrame(dst)
    return dst


def calc_ball_aggs(ball_pos, periods: List = [5], prod_periods: List = []):
    target_cols = [f"ball_{c}" for c in ["conf", "x", "y", "is_nan"]]
    target_cols += [f"ball_prod_p{p}" for p in prod_periods]
    target_cols += [f"ball_{xy}_diff_p{p}" for p in prod_periods for xy in "xy"]

    dst = []
    for p in periods:
        _dst = (
            ball_pos[target_cols]
            .rolling(p, center=True)
            .agg(["mean", "sum", "std", "min", "max", "median"])
        )
        _dst.columns = ["_".join(c) + f"_p{p}" for c in _dst.columns]
        dst.append(_dst)
    dst = pd.concat(dst, axis=1)
    return dst


def create_feature(video_filepath):
    video_id = os.path.basename(video_filepath).split(".")[0]

    save_dir = f"./data/feature/{video_id}"
    os.makedirs(save_dir, exist_ok=True)

    # NOTE: Save detect.csv.
    # obj_pos = detect_objct_position(video_filepath)
    # obj_pos.to_pickle(os.path.join(save_dir, "detect.pickle"))
    obj_pos = pd.read_csv(os.path.join(save_dir, "detect.csv"))

    # NOTE: Save ball.csv, only max confidence rows.
    ball_pos = ball(obj_pos)
    ball_pos.to_pickle(os.path.join(save_dir, "ball.pickle"))

    # NOTE: Save person.csv, only the 24 closest to the ball.
    person_pos = person(obj_pos, ball_pos)
    person_pos.to_pickle(os.path.join(save_dir, "person.pickle"))

    # NOTE: Save ball_movement.csv, diff some frames.
    periods = [1, 4, 8, 12, 16, 20, -1, -4, -8, -12, -16, -20]
    ball_movement = calc_movement(ball_pos, periods)
    ball_movement.to_pickle(os.path.join(save_dir, "ball_movement.pickle"))

    # NOTE: Save person_movement.csv, diff some frames.
    person_movement = calc_movement(person_pos, periods)
    person_movement.to_pickle(os.path.join(save_dir, "person_movement.pickle"))

    ball_prod = calc_arccos(ball_pos, ball_movement, periods)
    ball_prod.to_pickle(os.path.join(save_dir, "ball_prod.pickle"))

    ball_aggs = calc_ball_aggs(
        pd.concat((ball_pos, ball_movement, ball_prod), axis=1), [5], periods
    )
    ball_aggs.to_pickle(os.path.join(save_dir, "ball_aggs.pickle"))

    features = pd.concat(
        [
            ball_pos,
            person_pos.iloc[:, 1:],
            ball_movement,
            # person_movement,
            ball_prod,
            ball_aggs,
        ],
        axis=1,
    )
    features.to_pickle(os.path.join(save_dir, "features.pickle"))


def main():

    video_base_dir = "./data/raw/train/"
    video_filepath_list = glob.glob(os.path.join(video_base_dir, "*.mp4"))

    for video_filepath in tqdm(video_filepath_list):
        create_feature(video_filepath)


if __name__ == "__main__":
    main()
