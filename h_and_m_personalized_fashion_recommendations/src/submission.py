from collections import Counter
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import HMDataset
from models.base import HMModel
from utils import load_pickle, timer


def predict(model, dataloader):
    preds = []
    is_cuda = next(model.parameters()).is_cuda

    model.eval()
    for batch in tqdm(dataloader):
        inputs, _ = batch

        if is_cuda:
            inputs = {key: val.cuda() for key, val in inputs.items()}

        pred = model(inputs)
        _, pred = torch.topk(pred, k=12, dim=1)
        pred = pred.detach().cpu()
        preds.append(pred)

    preds = torch.cat(preds, dim=0)
    return preds


def get_top12_from_array(a: List[List]):
    top = []
    for _a in a:
        values, counts = zip(*Counter(a[0]).most_common(12))
        top.append(values)
    return top


def idx2articleId(x):
    articleIds_index = load_pickle("../data/working/article_id_map.pkl")
    index_articleIds = {v: k for k, v in articleIds_index.items()}

    resuts = []
    for _x in x:
        res = [index_articleIds[idx] for idx in _x]
        resuts.append(" ".join(res))
    return resuts


def main():
    config = Config()
    exp_num = config.exp_num

    eval_dataset = HMDataset(
        article_id_filepath="../data/feature/submission/article_id_seq.pkl",
        weeks_before_filepath="../data/feature/submission/weeks_before_seq.pkl",
        max_seq_len=config.max_seq_len,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=256,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    num_splits = 5
    preds = []
    for num_fold in range(num_splits):
        model = HMModel(
            article_embedding_size=config.article_embedding_size,
            max_seq_len=config.max_seq_len,
        )
        model.load_state_dict(
            torch.load(f"../data/model/{exp_num}/{num_fold}-fold.pth"),
        )
        model.to(torch.device("cuda"))

        pred = predict(model, eval_dataloader)
        preds.append(pred)

    preds = torch.cat(preds, dim=1)
    preds = preds.tolist()
    pred_ids = get_top12_from_array(preds)
    pred_article_ids = idx2articleId(pred_ids)

    sample_submission = pd.read_csv("../data/raw/sample_submission.csv")
    sample_submission["prediction"] = pred_article_ids
    sample_submission.to_csv(f"../data/submit/{exp_num}.csv", index=False)
    print(sample_submission.head())


if __name__ == "__main__":
    with timer("Submission"):
        main()
