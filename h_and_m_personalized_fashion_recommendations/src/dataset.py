import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.base import HMModel
from models.metric import BCEDiceLoss
from utils import load_pickle, mean_average_precision

articleId_index = load_pickle("../data/working/article_id_map.pkl")
DEBUG = True


class HMDataset(Dataset):
    def __init__(
        self,
        transaction_seq_filepath: str,
        customer_feat_filepath: str,
        weeks_before_filepath: str,
        target_filepath: str = None,
        max_seq_len: int = 32,
    ) -> None:
        # self.device = ("cuda" if torch.)
        self.transaction_seq = self._load_pickle(transaction_seq_filepath)
        self.customer_feat = self._load_pickle(customer_feat_filepath)
        # NOTE: Max value of weeks_before is 60.
        self.weeks_before = self._load_pickle(weeks_before_filepath)
        self.max_seq_len = max_seq_len

        if target_filepath is None:
            self.is_test = True
            self.target = np.zeros(shape=self.transaction_seq.shape[0])
        else:
            self.is_test = False
            self.target = self._load_pickle(target_filepath)["article_id"].to_numpy()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.is_test:
            target = self.target[idx]
        else:
            target = self._convert_multi_onehot(self.target[idx])

        article_id, ch_id, article_freq, token_id = self.transaction_seq[idx]
        customer_feat = torch.from_numpy(self.customer_feat[idx])

        article_id_seq = self._pad_sequence(article_id, dtype=torch.long)
        ch_id = self._pad_sequence(ch_id, dtype=torch.long)
        article_freq = self._pad_sequence(article_freq, dtype=torch.long)
        token_id = self._pad_sequence(token_id, dtype=torch.long)

        mask = (article_id_seq == 0).to(torch.bool)

        return {
            "mask": mask,
            "article_id_seq": article_id_seq,
            "channel_id_seq": ch_id,
            "article_id_freq_seq": article_freq,
            "active_token_id": token_id,
            "customer_feat": customer_feat,
        }, target

    def _load_pickle(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    def _pad_sequence(self, seq, dtype=torch.long):
        padded_seq = torch.zeros(self.max_seq_len, dtype=dtype)
        if isinstance(seq, list):
            seq = np.array(seq)
            if len(seq) >= self.max_seq_len:
                padded_seq = torch.from_numpy(seq[-self.max_seq_len :])
            else:
                # TODO: 左詰めにして、右側をpad
                padded_seq[-len(seq) :] = torch.from_numpy(seq)
        return padded_seq.to(dtype)

    def _convert_multi_onehot(self, targets: Dict):
        num_article_id = len(articleId_index)

        target = torch.zeros(num_article_id, dtype=torch.float32)
        for key, val in targets.items():
            target[key - 1] = val
        return target


def get_dataloaders(num_fold: int, batch_size: int = 256, max_seq_len: int = 16):
    train_dataset = HMDataset(
        target_filepath=f"../data/feature/{num_fold}_train_y.pkl",
        transaction_seq_filepath=f"../data/feature/{num_fold}_train/transaction_seq.pkl",
        customer_feat_filepath=f"../data/feature/{num_fold}_train/customer_feat.pkl",
        weeks_before_filepath=f"../data/feature/{num_fold}_train/weeks_before_seq.pkl",
        max_seq_len=max_seq_len,
    )
    valid_dataset = HMDataset(
        target_filepath=f"../data/feature/{num_fold}_valid_y.pkl",
        transaction_seq_filepath=f"../data/feature/{num_fold}_valid/transaction_seq.pkl",
        customer_feat_filepath=f"../data/feature/{num_fold}_valid/customer_feat.pkl",
        weeks_before_filepath=f"../data/feature/{num_fold}_valid/weeks_before_seq.pkl",
        max_seq_len=max_seq_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    max_seq_len = 16

    num_fold = 0
    dataset = HMDataset(
        target_filepath=f"../data/feature/{num_fold}_train_y.pkl",
        transaction_seq_filepath=f"../data/feature/{num_fold}_train/transaction_seq.pkl",
        customer_feat_filepath=f"../data/feature/{num_fold}_train/customer_feat.pkl",
        weeks_before_filepath=f"../data/feature/{num_fold}_train/weeks_before_seq.pkl",
        max_seq_len=max_seq_len,
    )

    dataloader = DataLoader(dataset, 4, False)
    inputs, target = next(iter(dataloader))
    print(inputs)

    model = HMModel(max_seq_len=max_seq_len)
    z = model(inputs)
    print(z.shape)

    critrion = BCEDiceLoss()
    loss = critrion(z, target)
    loss.backward()
    print(loss)

    _, pred = torch.topk(z, k=12, dim=1)
    pred = pred.detach().cpu().numpy()

    target = [t.nonzero().view(-1).detach().cpu().tolist() for t in target]

    targets = np.array(target, dtype=object)
    preds = np.array(pred)

    metric = mean_average_precision(targets, preds, k=12)
    print(metric)
