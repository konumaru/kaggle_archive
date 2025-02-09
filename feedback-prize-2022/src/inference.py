import glob
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from cuml import SVR
from cuml.ensemble import RandomForestRegressor
from transformers import AutoTokenizer

from config import (
    DeBERTaBase256Config,
    DeBERTaBase768Config,
    DeBERTaBaseConfig,
    DeBERTaLarge256Config,
    DeBERTaLargeConfig,
    DeBERTaV3Base256Config,
    DeBERTaV3Base768Config,
    DeBERTaV3BaseConfig,
    DeBERTaV3Large256Config,
    DeBERTaV3LargeConfig,
    RoBERTaBase256Config,
    RoBERTaBaseConfig,
    RoBERTaLargeConfig,
)
from dataset import get_inference_dataloader
from model import FeedbackModel


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def predict(model, dataloader, device):
    preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output, _ = model(input_ids, attention_mask)
            output = output.detach()

            preds.append(output.cpu().numpy())

    return np.concatenate(preds, axis=0)


def pred_stack_model(model_name, inputs, target_cols):

    preds = np.zeros((len(inputs), len(target_cols)))
    for i, t in enumerate(target_cols):
        model_path = list(
            glob.glob(f"../data/working/stacking/models_{model_name}/{t}_*.pkl")
        )
        models = [load_pickle(filepath) for filepath in model_path]
        preds[:, i] = np.mean([m.predict(inputs) for m in models], axis=0)

    return preds


def main():
    work_dir = pathlib.Path("../data/raw/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_cols = [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ]

    test = pd.read_csv(work_dir / "test.csv")

    configs = [
        # DeBERTaBase256Config,
        DeBERTaBase768Config,
        DeBERTaBaseConfig,
        DeBERTaLarge256Config,
        DeBERTaLargeConfig,
        DeBERTaV3Base256Config,
        DeBERTaV3Base768Config,
        DeBERTaV3BaseConfig,
        DeBERTaV3Large256Config,
        DeBERTaV3LargeConfig,
        RoBERTaBase256Config,
        RoBERTaBaseConfig,
        # RoBERTaLargeConfig,
    ]
    preds = pd.DataFrame(index=test["text_id"])

    for config in configs:
        exp_name = config.exp_name
        model_path = config.model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        test_dataloader = get_inference_dataloader(
            test, tokenizer, config.batch_size, config.max_seq_len
        )

        model_dir = pathlib.Path(f"../data/working/{exp_name}")
        model_path_list = glob.glob(str(model_dir / "*.pth"))
        model = FeedbackModel(model_path).to(device)

        single_preds = np.zeros((len(test), len(target_cols)))
        for model_path in model_path_list:
            model.load_state_dict(torch.load(model_path, map_location=device))
            single_preds += predict(model, test_dataloader, device)

        single_preds = single_preds / len(model_path_list)

        for i, t in enumerate(target_cols):
            preds[f"{t}_{exp_name}"] = single_preds[:, i]

    print(preds.head())
    preds.to_csv("../data/preds.csv")

    preds_svr = pred_stack_model("svr", preds, target_cols)
    preds_rf = pred_stack_model("rf", preds, target_cols)

    submission = pd.DataFrame(index=test["text_id"])
    submission[target_cols] = (preds_svr + preds_rf) / 2.0

    submission[target_cols] = submission[target_cols].clip(1.0, 5.0)
    submission.to_csv("../data/submission.csv")

    print(submission.head())


if __name__ == "__main__":
    main()
