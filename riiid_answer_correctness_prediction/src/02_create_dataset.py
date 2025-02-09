import os
import gc
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

from utils.common import timer
from saint_model_with_torch import SAINTModel


def add_question_data(data):
    questions = pd.read_csv("../data/raw/questions.csv")
    q_part_ids_map = dict(zip(questions["question_id"], questions["part"]))
    data["part"] = data["content_id"].map(q_part_ids_map).astype(np.int64)
    return data


def preprocessing(data):
    data = data[data["content_type_id"] == 0]
    data = data[["row_id", "user_id", "content_id", "answered_correctly"]]
    data = add_question_data(data)

    data = data.groupby("user_id").apply(
        lambda row: (
            row["content_id"].values,
            row["part"].values,
            row["answered_correctly"].values,
        )
    )
    # Drop <= 5 questions answered.
    data = data[data.apply(lambda x: x[0].shape[0]) > 5]
    return data


class SAINTDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_seq=100):
        self.user_ids = []
        self.df = df
        self.max_seq = max_seq
        for user_id in df.index.values:
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        (q_, c_, r_) = self.df[user_id]
        seq_len = len(q_)

        q_ = torch.as_tensor(q_, dtype=int)
        c_ = torch.as_tensor(c_, dtype=int)
        r_ = torch.as_tensor(r_, dtype=int)

        q = torch.zeros(self.max_seq, dtype=int)
        c = torch.zeros(self.max_seq, dtype=int)
        r = torch.zeros(self.max_seq, dtype=int)
        y = torch.zeros(self.max_seq, dtype=int)

        src_mask = torch.ones(self.max_seq, dtype=bool)
        label_mask = torch.ones(self.max_seq, dtype=bool)

        src_mask[:seq_len] = False
        label_mask[:seq_len] = False

        r[0] = 2  # 2-for the start of the sequence
        if seq_len > self.max_seq:
            q[:] = q_[: self.max_seq]
            c[:] = c_[: self.max_seq]
            r[1:] = r_[: self.max_seq - 1]
            y[:] = r_[: self.max_seq]
        elif seq_len <= self.max_seq:
            q[:seq_len] = q_
            c[:seq_len] = c_
            r[1:seq_len] = r_[: seq_len - 1]
            y[:seq_len] = r_

        return (q, c, r, y, src_mask, label_mask)


def main():
    src_dir = "../data/01_split/"
    dst_dir = "../data/02_create_dataset/"

    num_fold = 5
    for n_fold in range(num_fold):
        for data_type in ["train", "valid"]:
            filepath = os.path.join(src_dir, f"fold_{n_fold}_{data_type}.parquet")
            data = pd.read_parquet(filepath)

            data = preprocessing(data)
            dataset = SAINTDataset(data)

            filepath = os.path.join(dst_dir, f"fold_{n_fold}_{data_type}.pth")
            torch.save(dataset, filepath)
            print(f"Dump file to {filepath}")


if __name__ == "__main__":
    main()
