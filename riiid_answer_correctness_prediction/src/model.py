import os
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from utils.common import timer
from dataset import get_dataloader


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps,
    and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]


class EncoderEmbedding(nn.Module):
    def __init__(self, n_content, n_part, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len

        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.content_embed = nn.Embedding(n_content, n_dims)
        self.part_embed = nn.Embedding(n_part, n_dims)

    def forward(self, content_id, part_id):
        seq = torch.arange(self.seq_len, device=device).unsqueeze(0)
        pos = self.position_embed(seq)

        content = self.content_embed(content_id)
        part = self.part_embed(part_id)
        return pos + content + part


class DecoderEmbedding(nn.Module):
    def __init__(self, n_response, n_elapsed_time, n_lag_time, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len

        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.response_embed = nn.Embedding(n_response, n_dims)
        self.elapsed_time_embed = nn.Embedding(n_elapsed_time, n_dims)
        self.lag_time_embed = nn.Embedding(n_lag_time, n_dims)

    def forward(self, response, elapsed_time, lag_time):
        seq = torch.arange(self.seq_len, device=device).unsqueeze(0)
        pos = self.position_embed(seq)

        res = self.response_embed(response)
        els_time = self.elapsed_time_embed(elapsed_time)
        l_time = self.lag_time_embed(lag_time)
        return pos + res + els_time + l_time


class SAINTModule(pl.LightningModule):
    def __init__(
        self,
        num_content_id: int = 13523 + 1,
        num_part_id: int = 7 + 1,
        num_response: int = 4 + 1,
        num_elapsed_time: int = 300 + 1,
        num_lag_time: int = 150 + 1,
        max_seq_len: int = 100,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_encode_layers: int = 4,
        num_dencode_layers: int = 4,
    ):
        super(SAINTModule, self).__init__()
        self.max_seq_len = max_seq_len
        self.criterion = nn.BCELoss()

        self.encoder_embedding = EncoderEmbedding(
            n_content=num_content_id,
            n_part=num_part_id,
            n_dims=d_model,
            seq_len=max_seq_len,
        )
        self.decoder_embedding = DecoderEmbedding(
            n_response=num_response,
            n_elapsed_time=num_elapsed_time,
            n_lag_time=num_lag_time,
            n_dims=d_model,
            seq_len=max_seq_len,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encode_layers,
            num_decoder_layers=num_dencode_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.fc1 = nn.Linear(in_features=d_model, out_features=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x, y, src_pad_mask=None, tgt_pad_mask=None):
        src_mask = self._generate_square_subsequent_mask(x["content_id"].size(0))
        tgt_mask = self._generate_square_subsequent_mask(y.size(0))

        enc = self.encoder_embedding(
            content_id=x["content_id"],
            part_id=x["part_id"],
        )
        dec = self.decoder_embedding(
            response=x["response"],
            elapsed_time=x["elapsed_time"],
            lag_time=x["lag_time"],
        )

        x = self.transformer(
            enc,
            dec,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            # MEMO: training_stepでmaskを活用しているのでここでは無視して良い
            # src_key_padding_mask=src_pad_mask.T,
            # tgt_key_padding_mask=tgt_pad_mask.T,
        )
        x2 = self.fc1(x)
        out = self.softmax(x2)
        return out

    def _generate_square_subsequent_mask(self, len_seq):
        mask = (
            (torch.triu(torch.ones(len_seq, len_seq)) == 1).transpose(0, 1).to(device)
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = NoamLR(optimizer, warmup_steps=4000)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_ids):
        x, labels = batch

        out = self(x, labels, x["src_pad_mask"], x["tgt_pad_mask"])
        out = out.reshape(-1, self.max_seq_len)
        out = torch.masked_select(out, torch.logical_not(x["tgt_pad_mask"]))
        labels = torch.masked_select(labels, torch.logical_not(x["tgt_pad_mask"]))
        loss = self.criterion(out.float(), labels.float())
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def training_epoch_end(self, outputs):
        out = np.concatenate(
            [i["outs"].cpu().detach().numpy() for i in outputs]
        ).reshape(-1)
        labels = np.concatenate(
            [i["labels"].cpu().detach().numpy() for i in outputs]
        ).reshape(-1)
        auc = roc_auc_score(labels, out)

        self.print(f"\ntrain auc {auc}\n")
        self.log("train_auc", auc)

    def validation_step(self, batch, batch_ids):
        x, labels = batch

        out = self(x, labels, x["src_pad_mask"], x["tgt_pad_mask"])
        out = out.reshape(-1, self.max_seq_len)
        out = torch.masked_select(out, torch.logical_not(x["tgt_pad_mask"]))
        labels = torch.masked_select(labels, torch.logical_not(x["tgt_pad_mask"]))
        loss = self.criterion(out.float(), labels.float())

        self.log("valid_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return {"valid_loss": loss, "outs": out, "labels": labels}

    def validation_epoch_end(self, outputs):
        out = np.concatenate(
            [i["outs"].cpu().detach().numpy() for i in outputs]
        ).reshape(-1)
        labels = np.concatenate(
            [i["labels"].cpu().detach().numpy() for i in outputs]
        ).reshape(-1)
        auc = roc_auc_score(labels, out)

        self.print(f"\nvalid auc {auc}\n")
        self.log("valid_auc", auc)


def test_model(x, y):
    for key, val in x.items():
        x[key] = val.to(device)
    y = y.to(device)

    saint = SAINTModule()
    saint.to(device)

    criterion = nn.BCELoss()

    z = saint(x, y, x["src_pad_mask"], x["tgt_pad_mask"])

    z = z.reshape(128, 100)
    z = torch.masked_select(z, torch.logical_not(x["tgt_pad_mask"]))
    y = torch.masked_select(y, torch.logical_not(x["tgt_pad_mask"]))
    loss = criterion(z.float(), y.float())
    loss.backward()
    print(loss)


def main():
    pl.seed_everything(42)

    train_path = "../data/01_split/fold_0_train.parquet"
    valid_path = "../data/01_split/fold_0_valid.parquet"

    with timer("Get dataloader"):
        train_dataloader, valid_dataloader = get_dataloader(train_path, valid_path)

    # Test ========================================

    # x, y = next(iter(train_dataloader))
    # test_model(x, y)

    # ==============================================

    saint = SAINTModule()
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=10, verbose=False, mode="max"
    )
    mlf_logger = MLFlowLogger(
        experiment_name="default",
        tracking_uri="file:./ml-runs",
        save_dir="./",
    )
    trainer = pl.Trainer(
        gpus=4,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        accelerator="ddp",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        logger=mlf_logger,
        fast_dev_run=False,
        limit_test_batches=1.0,
        limit_val_batches=1.0,
    )
    trainer.fit(
        model=saint,
        train_dataloader=train_dataloader,
        val_dataloaders=[
            valid_dataloader,
        ],
    )


if __name__ == "__main__":
    with timer("Training"):
        main()
