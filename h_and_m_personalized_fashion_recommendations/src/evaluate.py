from collections import Counter
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import HMDataset
from models.base import HMModel
from utils import load_pickle, mean_average_precision, save_pickle


def predict(model, dataloader):
    preds = []
    use_cuda = next(model.parameters()).is_cuda

    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        inputs, _ = batch

        if use_cuda:
            inputs = {key: val.cuda() for key, val in inputs.items()}

        pred = model(inputs).to(torch.float16)
        _, pred = torch.topk(pred, k=12, dim=1)
        pred = pred.detach().cpu().clone()
        preds.append(pred)

    preds = torch.cat(preds, dim=0)

    return preds


def get_top12_from_array(a: List[List]):
    top = []
    for _a in a:
        values, counts = zip(*Counter(_a).most_common(12))
        values = tuple(v + 1 for v in values)
        top.append(values)
    return top


def main():
    config = Config()
    exp_num = config.exp_num

    eval_dataset = HMDataset(
        target_filepath="../data/feature/evaluation_y.pkl",
        transaction_seq_filepath="../data/feature/evaluation/transaction_seq.pkl",
        customer_feat_filepath="../data/feature/evaluation/customer_feat.pkl",
        weeks_before_filepath="../data/feature/evaluation/weeks_before_seq.pkl",
        max_seq_len=config.max_seq_len,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=256,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    targets = load_pickle("../data/feature/evaluation_y.pkl")
    targets = targets["article_id"].apply(lambda x: list(x.keys()))

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
    preds = preds.cpu().tolist()
    pred_ids = get_top12_from_array(preds)

    save_pickle("tmp.pkl", pred_ids)
    metric = mean_average_precision(targets, pred_ids)
    print("Mean Average Precision :", metric)

    with open(f"../data/evaluation/{exp_num}-{metric:.6f}.txt", "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
