import os
import gc
from collections import deque
from collections import defaultdict

import numpy as np
import pandas as pd

import torch

from saint_model import SAINTModel


def strList_to_intList(x, split_str=","):
    if x is np.nan or x == "[]":
        return list()
    else:
        x = x.split(split_str)
        x = [int(_x) for _x in x]
        return x


def get_question_data():
    questions = pd.read_csv("../data/raw/questions.csv")
    q_part_ids_map = dict(zip(questions["question_id"], questions["part"]))
    return q_part_ids_map


def init_dict():
    return {
        "content_id": deque(maxlen=100),
        "part": deque(maxlen=100),
    }


class SAINTTestDataset(torch.utils.data.Dataset):
    def __init__(self, test, user_state, max_seq=100):
        super(SAINTTestDataset).__init__()
        self.test = test
        self.user_state = user_state
        self.max_seq = max_seq

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test.iloc[idx]

        q = torch.zeros(self.max_seq, dtype=int)
        c = torch.zeros(self.max_seq, dtype=int)
        r = torch.zeros(self.max_seq, dtype=int)
        y = torch.zeros(self.max_seq, dtype=int)
        label_mask = torch.ones(self.max_seq, dtype=bool)

        user_id = data["user_id"]
        seq_len = len(self.user_state[user_id]["content_id"])
        if user_id in self.user_state.keys():
            q[1 : seq_len + 1] = torch.as_tensor(self.user_state[user_id]["content_id"])
            c[1 : seq_len + 1] = torch.as_tensor(self.user_state[user_id]["part"])
        else:
            q[0] = torch.as_tensor(data["content_id"].tolsit())
            c[0] = torch.as_tensor(data["part"].tolsit())

        if data["prior_group_answers_correct"] is np.nan or "[]":
            _r = list()
        else:
            _r = eval(data["prior_group_answers_correct"])

        r[0] = 2
        r[1 : len(_r) + 1] = torch.as_tensor(_r)
        return (q, c, r, y, label_mask)


def main():
    src_dir = "../data/02_create_feature/"
    dst_dir = "../data/03_split/"

    # Define paramteres.
    n_questions = 13523
    n_categories = 8
    n_responses = 3

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAINTModel(n_questions, n_categories, n_responses, device=device)
    model.load_state_dict(torch.load("saint5.pth"))
    model.to(device)

    q_part_ids_map = get_question_data()
    user_state = defaultdict(init_dict)

    smpl_test = pd.read_csv("../data/raw/example_test.csv")
    for group, test in smpl_test.groupby("group_num"):
        sample_prediction_df = test.loc[test["content_type_id"] == 0, ["row_id"]].copy()
        # test = preprocessing(test, q_part_ids_map)
        test = test[test["content_type_id"] == 0]
        test["part"] = test["content_id"].map(q_part_ids_map).astype(np.int64)

        dataset = SAINTTestDataset(test, user_state)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

        for (q, c, r, y, label_mask) in dataloader:
            q = q.to(device)
            c = c.to(device)
            r = r.to(device)
            yout = model(q, c, r).float()

            sample_prediction_df["answered_correctly"] = (
                yout[:, -1].detach().cpu().clone().numpy()
            )
            print(sample_prediction_df.shape)
            print(sample_prediction_df.tail())

        # update user state
        grouped_test = test.groupby("user_id").apply(
            lambda x: (
                x["content_id"].cumsum().tolist(),
                x["part"].cumsum().tolist(),
            )
        )
        for user_id in grouped_test.index:
            u_data = grouped_test[user_id]
            if user_id in user_state.keys():
                user_state[user_id]["content_id"].extend(u_data[0])
                user_state[user_id]["part"].extend(u_data[1])
            else:
                user_state[user_id]["content_id"] = u_data[0]
                user_state[user_id]["part"] = u_data[1]


if __name__ == "__main__":
    main()
