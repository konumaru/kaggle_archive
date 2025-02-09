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


# ================================================


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


# =======================================================


def main():
    # train_path = "../data/01_split/fold_0_train.parquet"
    # valid_path = "../data/01_split/fold_0_valid.parquet"

    # train = pd.read_parquet(train_path)
    # train = preprocessing(train)

    # valid = pd.read_parquet(valid_path)
    # valid = preprocessing(valid)

    # print(train.head())

    # # Define dataloader
    batch_size = 256
    # train_dataset = SAINTDataset(train)
    # torch.save(train_dataset, "train_dataset.pth")
    train_dataset = torch.load("train_dataset.pth")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # val_dataset = SAINTDataset(valid)
    # torch.save(val_dataset, "val_dataset.pth")
    val_dataset = torch.load("val_dataset.pth")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # Define paramteres.
    n_questions = 13523
    n_categories = 8
    n_responses = 3

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAINTModel(n_questions, n_categories, n_responses, device=device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    def train_epoch():
        train_loss = []
        model.train()

        for i, (q, c, r, y, src_mask, label_mask) in enumerate(train_dataloader):
            q = q.to(device)
            c = c.to(device)
            r = r.to(device)
            y = y.to(device)
            src_mask = src_mask.to(device)
            label_mask = label_mask.to(device)

            optimizer.zero_grad()
            yout = model(q, c, r, src_mask, label_mask)

            yout = torch.masked_select(yout, torch.logical_not(label_mask))
            y = torch.masked_select(y, torch.logical_not(label_mask))

            yout = yout.float()
            y = y.float()

            loss_ = criterion(yout, y)
            loss_.backward()
            optimizer.step()
            train_loss.append(loss_.item())

        return np.mean(train_loss)

    def val_epoch():
        val_loss = []
        model.eval()

        with torch.no_grad():
            for (q, c, r, y, src_mask, label_mask) in val_dataloader:
                q = q.to(device)
                c = c.to(device)
                r = r.to(device)
                y = y.to(device)
                src_mask = src_mask.to(device)
                label_mask = label_mask.to(device)
                yout = model(q, c, r, src_mask, label_mask)

                yout = torch.masked_select(yout, torch.logical_not(label_mask))
                y = torch.masked_select(y, torch.logical_not(label_mask))

                yout = yout.float()
                y = y.float()

                loss_ = criterion(yout, y)
                val_loss.append(loss_.item())

        return np.mean(val_loss)

    num_epochs = 100
    best_score = None
    for i in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch()
        val_loss = val_epoch()

        epoch_end = time.time()
        print("Time To Run Epoch:{}".format((epoch_end - epoch_start) / 60))
        print(
            "Epoch:{} | Train Loss: {:.8f} | Val Loss:{:.8f}".format(
                i, train_loss, val_loss
            )
        )

        if (best_score is None) or (best_score > val_loss):
            best_score = val_loss
            torch.save(model.state_dict(), "saint{}.pth".format(i))
        gc.collect()


if __name__ == "__main__":
    with timer("SAINT"):
        main()
