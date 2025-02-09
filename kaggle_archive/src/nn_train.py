import os
import pathlib

import hydra
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle, save_txt


class UmGameRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(  # type: ignore
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        return optimizer


class UmGameDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = X.astype("float32").to_numpy()
        self.y = y.astype("float32").to_numpy()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class UmGameDataModule(L.LightningDataModule):
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_valid: pd.DataFrame,
        batch_size: int = 1024,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = UmGameDataset(self.X_train, self.y_train)
        self.valid_dataset = UmGameDataset(self.X_valid, self.y_valid)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    seed: int,
    save_filepath: str,
) -> np.ndarray:
    torch.manual_seed(seed)

    data_module = UmGameDataModule(
        X_train,
        y_train,
        X_valid,
        y_valid,
        batch_size=512,
    )
    model = UmGameRegressor(X_train.shape[1])
    lit_model = LitModel(model, learning_rate=1e-3)

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name="my_model"),
        callbacks=[checkpoint],
        max_epochs=25,
    )
    trainer.fit(lit_model, data_module)

    if save_filepath:
        torch.save(lit_model.model.state_dict(), save_filepath + ".pth")

    valid_dataloader = data_module.val_dataloader()

    preds = []
    model.cpu()
    model.eval()
    for batch in valid_dataloader:
        x, y = batch
        y_hat = model(x)
        preds.append(y_hat.cpu().detach().numpy().flatten())

    return np.concatenate(preds, axis=0).flatten()


def train(cfg: DictConfig) -> None:
    model_dir_suffix = f"nn/seed={cfg.seed}/"
    save_dir = pathlib.Path(cfg.path.train) / model_dir_suffix
    save_dir.mkdir(exist_ok=True, parents=True)

    feature = load_feature(cfg.path.feature, sorted(cfg.feature_names))
    print("Feature shape:", feature.shape)
    target: pl.DataFrame = load_pickle(
        f"{cfg.path.feature}/{cfg.target_name}.pkl"
    )
    fold: pl.DataFrame = load_pickle(f"{cfg.path.feature}/{cfg.fold_name}.pkl")

    oof = np.zeros(len(target))

    for i in range(5):
        print(f"Fold {i}")

        is_valid = fold["fold"].eq(i).alias("is_valid")

        X_train = feature.filter(~is_valid).to_pandas()
        y_train = target.filter(~is_valid).to_pandas()
        X_valid = feature.filter(is_valid).to_pandas()
        y_valid = target.filter(is_valid).to_pandas()

        oof[is_valid] = fit_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            seed=42,
            save_filepath=str(save_dir / str(i)),
        )

    save_pickle(str(save_dir / "oof.pkl"), oof)


def evaluate(cfg: DictConfig) -> None:
    model_dir_suffix = "nn/seed=42/"
    save_dir = pathlib.Path(cfg.path.train) / model_dir_suffix

    oof = load_pickle(str(save_dir / "oof.pkl"))
    target = load_pickle(f"{cfg.path.feature}/{cfg.target_name}.pkl")
    fold: pl.DataFrame = load_pickle(f"{cfg.path.feature}/{cfg.fold_name}.pkl")

    print("")
    scores = []
    for i in range(fold.n_unique()):
        is_valid = fold["fold"].eq(i).alias("is_valid")
        _target = target.filter(is_valid).to_numpy()
        _oof = oof[is_valid]
        score = mean_squared_error(_target, _oof, squared=False)
        scores.append(score)

        print(f"Fold {i}: {score}")

    score = np.mean(scores)
    print(f"\n\nRMSE: {score}")
    save_txt(
        str(save_dir / f"score_{score:.8f}.txt"),
        str(score),
    )


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
