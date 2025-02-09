import os
import gc

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch

from saint import preprocessing, SAINTDataset
from saint_model_with_torch import SAINTModel


def main():
    src_dir = "../data/02_create_feature/"
    dst_dir = "../data/03_split/"

    valid_path = "../data/01_split/evaluation.parquet"
    valid = pd.read_parquet(valid_path)
    valid = preprocessing(valid)
    val_dataset = SAINTDataset(valid)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
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
    model.load_state_dict(torch.load("saint3.pth"))
    model.to(device)

    pred = []
    label = []

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

            yout = yout.float().cpu().numpy()
            y = y.float().cpu().numpy()

            pred.append(yout)
            label.append(y)

    pred = np.concatenate(pred).ravel()
    label = np.concatenate(label).ravel()

    auc = roc_auc_score(label, pred)
    print(auc)


if __name__ == "__main__":
    main()
