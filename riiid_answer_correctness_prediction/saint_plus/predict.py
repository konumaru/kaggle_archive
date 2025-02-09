from config import Config
from dataset import get_dataloaders

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from collections import deque
from collections import defaultdict

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from train import PlusSAINTModule


def strList_to_intList(x, split_str=","):
    if x is np.nan or x == "[]":
        return list()
    else:
        x = x.split(split_str)
        x = [int(_x) for _x in x]
        return x


class UserState:
    def __init__(self):
        self.state = defaultdict(self._init_dict)

    def _init_dict(self):
        LAST_N = 100
        PAD = 0
        return {
            "user_id": int,
            "content_id": deque([PAD] * LAST_N, maxlen=LAST_N),
            "task_container_id": deque([PAD] * LAST_N, maxlen=LAST_N),
            # "part_id": deque([PAD] * LAST_N, maxlen=LAST_N),
            "prior_question_elapsed_time": deque([PAD] * LAST_N, maxlen=LAST_N),
            "padded": deque([False] * 100, maxlen=LAST_N),
            "answered_correctly": deque([0] * LAST_N, maxlen=LAST_N),
        }

    def update_state(self, row):
        user_id = row["user_id"]

        self.state[user_id]["user_id"] = row["user_id"]
        self.state[user_id]["content_id"].appendleft(row["content_id"])
        self.state[user_id]["task_container_id"].appendleft(row["task_container_id"])
        # self.state[user_id]["part_id"].appendleft(row["part_id"])
        self.state[user_id]["prior_question_elapsed_time"].appendleft(
            row["prior_question_elapsed_time"]
        )
        if len(row["prior_group_answers_correct"]) > 0:
            self.state[user_id]["answered_correctly"].extendleft(
                row["prior_group_answers_correct"]
            )

    def get_feature(self, row):
        # return [row["content_type_id"], self.state[row["user_id"]]]
        user_id = row["user_id"]
        return (
            list(self.state[user_id]["content_id"]),
            list(self.state[user_id]["prior_question_elapsed_time"]),
            list(self.state[user_id]["task_container_id"]),
            list(self.state[user_id]["answered_correctly"]),
        )


def preprocessing(test):
    test = test[test.content_type_id == 0]
    test.prior_question_elapsed_time.fillna(0, inplace=True)
    test.prior_question_elapsed_time /= 1000
    # test.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    test.prior_question_elapsed_time = test.prior_question_elapsed_time.astype(np.int)
    columns = ["prior_group_answers_correct", "prior_group_responses"]
    for c in columns:
        test[c] = test[c].str.extract("\[(.+)\]")
        test[c] = test[c].map(strList_to_intList)
    return test


def create_feature(data):
    def get_feature_and_update_state(us, idx, row):
        feature = us.get_feature(row)
        us.update_state(row)
        return feature

    us = UserState()

    result = Parallel(n_jobs=-1)(
        delayed(get_feature_and_update_state)(us, idx, row)
        for idx, row in data.iterrows()
    )
    result = np.array(result)
    return result


def main():
    device = Config.device
    modelpath = "./lightning_logs/version_1/checkpoints/epoch=3-step=22443.ckpt"
    model = PlusSAINTModule.load_from_checkpoint(
        modelpath,
        on_gpu=True,
    )
    model.to(device)
    model.eval()
    model.freeze()

    df = pd.read_csv("../data/raw/example_test.csv")
    for i, gdf in df.groupby("group_num"):
        test = gdf.set_index("group_num").copy()
        sample_prediction_df = test.loc[test["content_type_id"] == 0, ["row_id"]].copy()
        sample_prediction_df["answered_correctly"] = 0.5

        test = preprocessing(test)
        print(test.head())

        data = create_feature(test)
        print(data[:, 0].shape)

        pred = model(
            {
                "input_ids": torch.from_numpy(data[:, 0]).to(device),
                "input_rtime": torch.from_numpy(data[:, 1]).to(device),
                "input_cat": torch.from_numpy(data[:, 2]).to(device),
            },
            torch.from_numpy(data[:, 3]).to(device),
        )
        pred = pred[:, 0]

        sample_prediction_df["answered_correctly"] = pred.cpu().clone().numpy()
        print(sample_prediction_df.head())

        # plt.figure()
        # sample_prediction_df["answered_correctly"].hist(bins=25)
        # plt.savefig(f"group{i}_hist.png")

        # sample_prediction_df.to_csv(f'group{i}_submission.csv', index=False)


if __name__ == "__main__":
    main()
