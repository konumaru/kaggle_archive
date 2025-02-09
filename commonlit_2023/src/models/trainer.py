import logging
import os
import pathlib
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class PytorchTrainer:
    def __init__(
        self,
        work_dir: Union[str, pathlib.Path],
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module,
        eval_metric: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cuda",
    ) -> None:
        self.work_dir = pathlib.Path(work_dir)
        self.best_model_path = self.work_dir / "best_model.pth"

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        log_filepath = self.work_dir / "train.log"

        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        logger = logging.getLogger("Training")
        logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(format)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        return logger

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.to(self.device)
        self.model.eval()

        preds = []
        targets = []
        for inputs, _targets in self.valid_dataloader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
            targets.append(_targets.detach().cpu())

        preds_all = torch.cat(preds, dim=0)
        targets_all = torch.cat(targets, dim=0)
        score = self.eval_metric(preds_all, targets_all)
        return score.item()

    def train_batch(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]
    ) -> float:
        inputs, targets = batch
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        max_epochs: int,
        every_eval_steps: int = 100,
        save_interval: str = "epoch",
    ) -> None:
        assert save_interval in [
            "epoch",
            "batch",
        ], "save_interval must be epoch or batch"

        self.model.to(self.device)
        self.model.train()

        best_eval_score = float("inf")
        total_steps = len(self.train_dataloader) * max_epochs
        eval_score = None
        train_global_step = 0

        progress_bar = tqdm(
            total=total_steps, desc="Training", dynamic_ncols=True
        )
        with logging_redirect_tqdm():
            for epoch in range(max_epochs):
                running_loss = 0.0

                for batch_idx, (inputs, targets) in enumerate(
                    self.train_dataloader
                ):
                    loss = self.train_batch((inputs, targets))
                    running_loss += loss
                    avg_loss = running_loss / (batch_idx + 1)

                    if (
                        save_interval == "batch"
                        and train_global_step % every_eval_steps == 0
                    ):
                        eval_score = self.evaluate()
                        if eval_score < best_eval_score:
                            best_eval_score = eval_score
                            self.save_model(self.best_model_path)

                    lognameValues = {
                        "Epoch": epoch,
                        "loss": round(avg_loss, 4),
                        "eval_score": round(eval_score, 4),  # type: ignore
                        "best_score": round(best_eval_score, 4),
                    }
                    progress_bar.set_postfix(lognameValues)

                    message = " | ".join(
                        [f"{k}={v}" for k, v in lognameValues.items()]
                    )
                    self.logger.info(message)

                    progress_bar.update(1)
                    train_global_step += 1

                if save_interval == "epoch":
                    eval_score = self.evaluate()
                    if eval_score < best_eval_score:
                        best_eval_score = eval_score
                        self.save_model(self.best_model_path)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()

        preds = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(inputs)
                preds.append(outputs.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def save_model(self, filename: Union[str, pathlib.Path]) -> None:
        torch.save(self.model.state_dict(), filename)

    def load_best_model(self) -> None:
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"Loaded best model from {self.best_model_path}")
