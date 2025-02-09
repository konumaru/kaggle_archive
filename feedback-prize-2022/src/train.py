import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoTokenizer

from config import RoBERTaBase768Config as Config
from dataset import get_train_dataloader, get_valid_dataloader
from model import FeedbackModel, get_optimizer_params
from utils import set_seed, timer


def train(model, dataloader, optimizer, scheduler, loss_fn, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batch_losses = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output, _ = model(input_ids, attention_mask)
            batch_loss = loss_fn(labels, output)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_losses.append(batch_loss.item())

    scheduler.step()

    return np.mean(batch_losses)


def validation(model, dataloader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    batch_losses = []
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output, _ = model(input_ids, attention_mask)
            output = output.detach()

            batch_loss = loss_fn(output, labels)
            batch_losses.append(batch_loss.item())

            preds.append(output.cpu().numpy())

    return np.mean(batch_losses), np.concatenate(preds, axis=0)


def compute_MCRMSE(y_true, y_pred):
    col_rmse = [
        mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        for i in range(y_true.shape[1])
    ]
    return np.mean(col_rmse)


def fit(data, target_cols, save_dir, config, seed=42):
    set_seed(seed)

    cv = MultilabelStratifiedKFold(
        n_splits=config.num_fold, shuffle=True, random_state=seed
    )
    for fold, (_, valid_idx) in enumerate(cv.split(data, data[target_cols])):
        data.loc[valid_idx, "fold"] = fold

    data["fold"] = data["fold"].astype(int)
    print(data["fold"].value_counts().sort_index(), "\n")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)

    oof = pd.DataFrame(index=data["text_id"])
    oof[target_cols] = 0.0

    for fold in range(config.num_fold):
        print(f"\n>>> Fold - {fold} Training")
        train_df = data[data["fold"] != fold].reset_index(drop=True)
        valid_df = data[data["fold"] == fold].reset_index(drop=True)

        train_dataloader = get_train_dataloader(
            train_df,
            tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
        )
        valid_dataloader = get_valid_dataloader(
            valid_df,
            tokenizer,
            max_seq_len=config.max_seq_len,
        )

        model = FeedbackModel(config.model_path)
        model.to(config.device)

        criterion = nn.HuberLoss(reduction="mean", delta=1.0)
        optimizer_parameters = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5, weight_decay=1e-2)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        best_loss = np.inf
        for epoch in range(config.num_epoch):
            start_time = time.time()
            print(f"Epoch-{epoch:02} ->", end="")

            train_loss = train(model, train_dataloader, optimizer, scheduler, criterion)
            print(f" train_loss: {train_loss:.6f}", end="")

            valid_loss, valid_preds = validation(model, valid_dataloader, criterion)
            valid_score = compute_MCRMSE(valid_df[target_cols].to_numpy(), valid_preds)
            print(
                f" valid_loss: {valid_loss:.6f} valid_score: {valid_score:.6f}", end=""
            )

            if best_loss > valid_loss:
                best_loss = valid_loss
                oof.loc[valid_df["text_id"], target_cols] = valid_preds
                torch.save(
                    model.state_dict(),
                    str(save_dir / f"model_seed{seed}_fold{fold}.pth"),
                )

            elapsed = time.time() - start_time
            print(f" time: {elapsed:.0f}s")

        del train_df, valid_df, model, optimizer

    oof.to_csv(str(save_dir / f"oof_seed{seed}.csv"))

    score = compute_MCRMSE(data[target_cols].to_numpy(), oof[target_cols].to_numpy())
    print(score)
    with open(str(save_dir / f"seed{seed}_score={score:.6f}"), "w") as f:
        f.write("")


def main():
    config = Config()
    raw_dir = pathlib.Path("../data/raw")
    work_dir = pathlib.Path(f"../data/working/{config.exp_name}/")

    os.makedirs(str(work_dir), exist_ok=True)

    data = pd.read_csv(raw_dir / "train.csv")
    target_cols = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]
    sub_seeds = [0, 31, 2022]

    print("=== Start Training ===\n")

    for sub_seed in sub_seeds:
        print(f"fit with seed={config.seed + sub_seed}")
        fit(
            data=data,
            target_cols=target_cols,
            seed=config.seed + sub_seed,
            config=config,
            save_dir=work_dir,
        )

    print("=== Evaluation ===\n")
    oof = pd.DataFrame(index=data["text_id"])
    oof[target_cols] = 0.0
    for sub_seed in sub_seeds:
        seed = config.seed + sub_seed
        _oof = pd.read_csv(str(work_dir / f"oof_seed{seed}.csv"))
        oof[target_cols] += _oof[target_cols].to_numpy() / len(sub_seeds)

    oof.to_csv(str(work_dir / "oof.csv"))

    score = compute_MCRMSE(data[target_cols].to_numpy(), oof[target_cols].to_numpy())
    print(score)
    with open(str(work_dir / f"score={score:.6f}"), "w") as f:
        f.write("")


if __name__ == "__main__":
    with timer("Training"):
        main()
