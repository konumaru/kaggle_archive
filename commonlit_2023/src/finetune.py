import os
import pathlib
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from metric import mcrmse
from models import CommonLitModel
from models.dataset import CommonLitDataset
from models.metrics import MCRMSELoss
from models.trainer import PytorchTrainer
from utils import seed_everything, timer
from utils.io import save_txt

warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset) -> np.ndarray:
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=8
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    preds = []
    for batch in dataloader:
        inputs = batch[0]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        z = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        preds.append(z.logits.detach().cpu().numpy())

    results = np.concatenate(preds, axis=0)
    return results


def get_optimizer(model: nn.Module):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "head" in n],
            "weight_decay": 0.0,
            "lr": 3e-4,
        }
    ]

    no_decay = ["bias", "LayerNorm.weight"]
    layers = [getattr(model, "model").embeddings] + list(
        getattr(model, "model").encoder.layer
    )
    for layer in layers[-4:]:
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.02,
                "lr": 1e-4,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": 1e-4,
            },
        ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters, lr=1e-5, weight_decay=0.02
    )


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    input_dir = pathlib.Path(cfg.path.preprocessed)
    model_dir = pathlib.Path(cfg.path.model) / "deberta-v3-base"

    data = pd.read_csv(input_dir / "train.csv")
    data[["pred_content", "pred_wording"]] = 0.0

    model_name = "microsoft/deberta-v3-base"
    # model_name = "microsoft/deberta-v3-large"

    for fold in range(cfg.n_splits):
        print(f"Fold: {fold}")

        model = CommonLitModel(model_name, num_labels=2)
        train_dataset = CommonLitDataset(
            data.query(f"fold!={fold}"), model_name, max_len=512
        )
        valid_dataset = CommonLitDataset(
            data.query(f"fold=={fold}"), model_name, max_len=512
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=4, shuffle=True, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=4, shuffle=False
        )
        optimizer = get_optimizer(model)

        trainer = PytorchTrainer(
            work_dir=str(model_dir / f"fold{fold}"),
            model=model,
            criterion=MCRMSELoss(),
            eval_metric=MCRMSELoss(),
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer,
        )

        trainer.train(
            max_epochs=4,
            save_interval="batch",
            every_eval_steps=20,
        )
        trainer.load_best_model()

        pred = trainer.predict(valid_dataloader)
        data.loc[data["fold"] == fold, ["pred_content", "pred_wording"]] = pred

    score = mcrmse(
        data[["content", "wording"]].to_numpy(),
        data[["pred_content", "pred_wording"]].to_numpy(),
    )
    print(f"Score: {score}")
    save_txt(
        str(model_dir / "score.txt"),
        str(score),
    )

    data[
        ["prompt_id", "student_id", "fold", "pred_content", "pred_wording"]
    ].to_csv(str(model_dir / "oof.csv"), index=False)


if __name__ == "__main__":
    with timer("main.py"):
        main()
