import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.common import timer

# Model ==================================


class SAINTDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_seq=100):
        self.user_ids = []
        self.df = df
        self.max_seq = max_seq
        for user_id in df.index.values:
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        (q_, c_, r_) = self.df[user_id]
        seq_len = len(q_)

        q_ = torch.as_tensor(q_, dtype=int)
        c_ = torch.as_tensor(c_, dtype=int)
        r_ = torch.as_tensor(r_, dtype=int)

        q = torch.zeros(self.max_seq, dtype=int)
        c = torch.zeros(self.max_seq, dtype=int)
        r = torch.zeros(self.max_seq, dtype=int)
        y = torch.zeros(self.max_seq, dtype=int)

        src_mask = torch.ones(self.max_seq, dtype=bool)
        label_mask = torch.ones(self.max_seq, dtype=bool)

        src_mask[:seq_len] = False
        label_mask[:seq_len] = False

        r[0] = 2  # 2-for the start of the sequence
        if seq_len > self.max_seq:
            q[:] = q_[: self.max_seq]
            c[:] = c_[: self.max_seq]
            r[1:] = r_[: self.max_seq - 1]
            y[:] = r_[: self.max_seq]
        elif seq_len <= self.max_seq:
            q[:seq_len] = q_
            c[:seq_len] = c_
            r[1:seq_len] = r_[: seq_len - 1]
            y[:seq_len] = r_

        return (q, c, r, y, src_mask, label_mask)


import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional.classification import auroc

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoamLR(_LRScheduler):
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
    def __init__(self, n_response, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len

        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.response_embed = nn.Embedding(n_response, n_dims)

    def forward(self, response):
        seq = torch.arange(self.seq_len, device=device).unsqueeze(0)
        pos = self.position_embed(seq)

        res = self.response_embed(response)
        return pos + res


class SAINTModel(pl.LightningModule):
    def __init__(
        self,
        n_questions,
        n_categories,
        n_responses,
        max_seq=100,
        d_model=512,
        encoder_dim=128,
        decoder_dim=128,
        num_heads=4,
    ):
        super(SAINTModel, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

        self.encoder_embedding = EncoderEmbedding(
            n_content=n_questions,
            n_part=n_categories,
            n_dims=d_model,
            seq_len=max_seq,
        )
        self.decoder_embedding = DecoderEmbedding(
            n_response=n_responses,
            n_dims=d_model,
            seq_len=max_seq,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
        )
        self.fc1 = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, q, c, r, src_pad_mask, tgt_pad_mask):
        enc = self.encoder_embedding(
            content_id=q,
            part_id=c,
        )
        dec = self.decoder_embedding(
            response=r,
        )
        x = self.transformer(
            enc,
            dec,
            src_key_padding_mask=src_pad_mask.T,
            tgt_key_padding_mask=tgt_pad_mask.T,
        )
        x = self.fc1(x)
        return x.squeeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = NoamLR(optimizer, warmup_steps=4000)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_ids):
        (q, c, r, y, src_mask, label_mask) = batch

        yout = self(q, c, r, src_mask, label_mask)

        out = torch.masked_select(yout, torch.logical_not(label_mask)).float()
        y = torch.masked_select(y, torch.logical_not(label_mask)).float()
        loss = self.criterion(out, y)

        proba = out.clone().view(-1)
        label = y.clone().view(-1)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"train_loss": loss, "proba": proba, "label": label}

    # def training_step_end(self, batch_parts):
    #     result = {}
    #     result["loss"] = torch.mean(torch.stack([x["train_loss"] for x in batch_parts]))
    #     result["proba"] = torch.stack([x["pred"] for x in batch_parts]).view(-1)
    #     result["label"] = torch.stack([x["label"] for x in batch_parts]).view(-1)

    #     # do something with both outputs
    #     return (batch_parts[0]["loss"] + batch_parts[1]["loss"]) / 2

    def training_epoch_end(self, outputs):
        probs = []
        labels = []
        for output in outputs:
            probs.append(output["proba"])
            labels.append(output["label"])

        proba = torch.cat(probs, dim=0)
        label = torch.cat(labels, dim=0)
        auc = auroc(proba, label)
        self.log("train_metric", auc)

    def validation_step(self, batch, batch_ids):
        (q, c, r, y, src_mask, label_mask) = batch

        yout = self(q, c, r, src_mask, label_mask)

        out = torch.masked_select(yout, torch.logical_not(label_mask))
        y = torch.masked_select(y, torch.logical_not(label_mask))
        out = out.float()
        y = y.float()
        loss = self.criterion(out, y)

        proba = out.clone().view(-1)
        label = y.clone().view(-1)
        self.log("valid_loss", loss, on_step=True, prog_bar=True)
        return {"valid_loss": loss, "proba": proba, "label": label}

    def validation_epoch_end(self, outputs):
        probs = []
        labels = []
        for output in outputs:
            probs.append(output["proba"])
            labels.append(output["label"])

        proba = torch.cat(probs, dim=0)
        label = torch.cat(labels, dim=0)
        auc = auroc(proba, label)
        self.log("valid_metric", auc)


# ========================================


def get_dataloader(train_path, valid_path, batch_size=256):
    train_dataset = torch.load(train_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_dataset = torch.load("val_dataset.pth")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader


def test_model(batch):
    (q, c, r, y, src_mask, label_mask) = batch

    n_questions = 13523
    n_categories = 8
    n_responses = 3
    saint = SAINTModel(n_questions, n_categories, n_responses)

    yout = saint(q, c, r, src_mask, label_mask)

    out = torch.masked_select(yout, torch.logical_not(label_mask))
    y = torch.masked_select(y, torch.logical_not(label_mask))
    out = out.float()
    y = y.float()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(out, y)
    print(loss)

    proba = out.clone().view(-1)
    label = y.clone().view(-1)

    proba_v2 = torch.cat((proba, proba), dim=0)
    label_v2 = torch.cat((label, label), dim=0)
    auc = auroc(proba_v2, label_v2)
    print(auc)


def main():
    num_fold = 1  # 5
    for i in range(num_fold):
        train_path = f"../data/02_create_dataset/fold_{i}_train.pth"
        valid_path = f"../data/02_create_dataset/fold_{i}_valid.pth"

        with timer("Get dataloader"):
            train_dataloader, valid_dataloader = get_dataloader(train_path, valid_path)

        # batch = next(iter(train_dataloader))
        # test_model(batch)

        n_questions = 13523
        n_categories = 8
        n_responses = 3
        saint = SAINTModel(n_questions, n_categories, n_responses)
        early_stop_callback = EarlyStopping(
            monitor="valid_loss", min_delta=0.00, patience=10, verbose=False, mode="max"
        )
        # mlf_logger = MLFlowLogger(
        #     experiment_name="default", tracking_uri="file:./ml-runs"
        # )
        trainer = pl.Trainer(
            gpus=4,
            max_epochs=50,
            progress_bar_refresh_rate=21,
            accelerator="ddp",
            # gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            # logger=mlf_logger,
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
    main()
