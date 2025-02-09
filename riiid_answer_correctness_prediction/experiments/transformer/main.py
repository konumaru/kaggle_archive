import gc
import math
import time
import pickle
import itertools
from tqdm import tqdm
from time import time
from pathlib import Path
from collections import deque
from collections import Counter
from collections import namedtuple
from contextlib import contextmanager

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
gc.enable()

TRAIN_DTYPES = {
    # 'row_id': np.uint32,
    "timestamp": np.uint64,
    "user_id": np.uint32,
    "content_id": np.uint16,
    "content_type_id": np.uint8,
    "task_container_id": np.uint16,
    "user_answer": np.int8,
    "answered_correctly": np.int8,
    "prior_question_elapsed_time": np.float32,
    "prior_question_had_explanation": "boolean",
}

DATA_DIR = Path("../../data/raw")
TRAIN_PATH = DATA_DIR / "train.csv"
QUESTIONS_PATH = DATA_DIR / "questions.csv"
LECTURES_PATH = DATA_DIR / "lectures.csv"

# this parameter denotes how many last seen content_ids
# I am going to consider <aka the max_seq_len or the window size>.
LAST_N = 100


class TransformerModel(nn.Module):
    def __init__(
        self,
        ninp: int = 32,
        nhead: int = 2,
        nhid: int = 64,
        nlayers: int = 2,
        dropout: float = 0.3,
    ):
        """
        nhead -> number of heads in the transformer multi attention thing.
        nhid -> the number of hidden dimension neurons in the model.
        nlayers -> how many layers we want to stack.
        """
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=nlayers
        )
        self.exercise_embeddings = nn.Embedding(
            num_embeddings=13523, embedding_dim=ninp
        )  # exercise_id
        self.pos_embedding = nn.Embedding(ninp, ninp)  # positional embeddings
        self.part_embeddings = nn.Embedding(
            num_embeddings=7 + 1, embedding_dim=ninp
        )  # part_id_embeddings
        self.prior_question_elapsed_time = nn.Embedding(
            num_embeddings=301, embedding_dim=ninp
        )  # prior_question_elapsed_time
        self.device = "cpu"
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # init embeddings
        self.exercise_embeddings.weight.data.uniform_(-initrange, initrange)
        self.part_embeddings.weight.data.uniform_(-initrange, initrange)
        self.prior_question_elapsed_time.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, content_id, part_id, prior_question_elapsed_time=None, mask_src=None
    ):
        """
        S is the sequence length,
        N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        """

        embedded_src = (
            self.exercise_embeddings(content_id)
            + self.pos_embedding(
                torch.arange(0, content_id.shape[1])
                .to(self.device)
                .unsqueeze(0)
                .repeat(content_id.shape[0], 1)
            )
            + self.part_embeddings(part_id)
            + self.prior_question_elapsed_time(prior_question_elapsed_time)
        )  # (N, S, E)
        embedded_src = embedded_src.transpose(0, 1)  # (S, N, E)

        _src = embedded_src * np.sqrt(self.ninp)

        output = self.transformer_encoder(src=_src, src_key_padding_mask=mask_src)
        output = self.decoder(output)
        output = output.transpose(1, 0)
        return output


def pad_seq(
    seq: List[int], max_batch_len: int = LAST_N, pad_value: int = True
) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]


class Riiid(torch.utils.data.Dataset):
    def __init__(self, d:Dict):
        self.d = d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        # you can return a dict of these as well etc etc...
        # remember the order
        return (
            idx,
            self.d[idx]["content_id"],
            self.d[idx]["task_container_id"],
            self.d[idx]["part_id"],
            self.d[idx]["prior_question_elapsed_time"],
            self.d[idx]["padded"],
            self.d[idx]["answered_correctly"],
        )


def collate_fn(batch):
    _, content_id, task_id, part_id, prior_question_elapsed_time, padded, labels = zip(
        *batch
    )
    content_id = torch.Tensor(content_id).long()
    task_id = torch.Tensor(task_id).long()
    part_id = torch.Tensor(part_id).long()
    prior_question_elapsed_time = torch.Tensor(prior_question_elapsed_time).long()
    padded = torch.Tensor(padded).bool()
    labels = torch.Tensor(labels)
    # remember the order
    return content_id, task_id, part_id, prior_question_elapsed_time, padded, labels

def get_deque_feature(feature: pd.Series, max_len: int=100, pad: int=0):
    if len(feature) > 100:
        d_feature = deque(feature, maxlen=max_len)
    else:
        num_pad = len(feature)
        d_feature = deque(feature + [pad] * (max_len - num_pad), maxlen=max_len)
    return d_feature

def main():
    '''Transform
    '''
    print('>'*5, 'Run Transform.\n')
    # Load data.
    df_questions = pd.read_csv(QUESTIONS_PATH)
    df_train = pd.read_csv(
        TRAIN_PATH, nrows=40_00_000, dtype=TRAIN_DTYPES, usecols=TRAIN_DTYPES.keys()
    )

    df_train["prior_question_had_explanation"] = (
        df_train["prior_question_had_explanation"]
        .astype(np.float16)
        .fillna(-1)
        .astype(np.int8)
    )
    df_train = df_train[df_train.content_type_id == 0]

    part_ids_map = dict(zip(df_questions.question_id, df_questions.part))
    df_train["part_id"] = df_train["content_id"].map(part_ids_map)

    df_train["prior_question_elapsed_time"].fillna(
        26000, inplace=True
    )  # some random value fill in
    df_train["prior_question_elapsed_time"] = (
        df_train["prior_question_elapsed_time"] // 1000
    )

    print(df_train.shape)
    print(df_train.head())

    '''Define Dataset
    '''
    print('>'*5, 'Run Define Dataset.\n')
    PAD = 0
    grp = df_train.groupby("user_id").tail(LAST_N)  # Select last_n rows of each user.

    d = {}
    user_id_to_idx = {}
    for idx, row in tqdm(
        grp.groupby("user_id")
        .agg(
            {
                "content_id": list,
                "answered_correctly": list,
                "task_container_id": list,
                "part_id": list,
                "prior_question_elapsed_time": list,
            }
        )
        .reset_index()
        .iterrows(),
        total=df_train["user_id"].nunique(),
    ):
        len_pad = min(len(row["content_id"]), 100)

        d[idx] = {
            "user_id": row["user_id"],
            # "content_id": deque(row["content_id"], maxlen=LAST_N),get_deque_feature
            "content_id": get_deque_feature(row["content_id"]),
            "answered_correctly": get_deque_feature(row["answered_correctly"]),
            "task_container_id": get_deque_feature(row["task_container_id"]),
            "prior_question_elapsed_time": get_deque_feature(
                row["prior_question_elapsed_time"]),
            "part_id": get_deque_feature(row["part_id"]),
            "padded": get_deque_feature([False] * len_pad),
        }
        user_id_to_idx[row["user_id"]] = idx

    # print(d[0])

    d_train, d_test = train_test_split(d, test_size=0.2)

    # Define pytorch dataset.
    dataset = Riiid(d=d_train)
    print(dataset[0])

    '''Train
    '''
    print('>'*5, 'Run Train.\n')
    model = TransformerModel(ninp=LAST_N, nhead=4, nhid=128, nlayers=3, dropout=0.3)
    # print(model)  # look into it!

    BATCH_SIZE = 32
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                collate_fn=collate_fn,
                num_workers=8,
            )

    model.train()
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode=True):
            output = model(content_id, part_id, prior_question_elapsed_time, mask)
            # output is (N,S,2) # i am working on it
            loss = criterion(output[:, :, 1], labels)
            loss.backward()
            losses.append(loss.detach().data.numpy())
            optimizer.step()

    plt.figure()
    pd.Series(losses).astype(np.float32).plot(kind="line")
    plt.savefig("loss.png")

    '''Evaluation
    '''
    dataset = Riiid(d=d_test)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        drop_last=True
    )

    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(valid_dataloader)):
            content_id, _, part_id, prior_question_elapsed_time, mask, labels = batch
            pred = model(content_id, part_id, prior_question_elapsed_time, mask)
            pred = pred[:, :, 1]

            if idx == 0:
                valid_preds = pred
                valid_labels = labels
            else:
                valid_preds = torch.cat([valid_preds, pred], axis=0)
                valid_labels = torch.cat([valid_labels, labels], axis=0)

    print(valid_preds)
    print(valid_labels)


    print('AUC:', roc_auc_score(
        torch.flatten(valid_labels).numpy(),
        torch.flatten(valid_preds).numpy()
    ))



if __name__ == "__main__":
    main()
