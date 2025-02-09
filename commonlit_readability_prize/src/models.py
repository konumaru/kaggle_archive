import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class CommonLitRoBERTaModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        output_hidden_states: bool = False,
    ):
        super(CommonLitRoBERTaModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=output_hidden_states,
        )
        self.config = self.roberta.config

        hidden_size = self.config.hidden_size
        self.att_head = AttentionHead(hidden_size, hidden_size, 1)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
        )
        # Initialize Weights
        self.att_head.apply(self._init_weights)
        self.regression_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        outputs = self.roberta(**batch["inputs"])
        # NOTE: この段階でtextstat特徴量を追加すると、fine-tuning時の学習率と相性が悪く過学習する可能性が高い、
        # x = torch.cat((pooler_output, batch["textstat"]), dim=1)
        # x = self.att_head(outputs.last_hidden_state)
        x = self.regression_head(outputs.pooler_output)
        return x


class CommonLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 5e-5,
        num_epochs: int = 10,
        roberta_model_name_or_path: str = "roberta-base",
        output_hidden_states: bool = False,
        lr_scheduler: str = "linear",
        lr_interval: str = "epoch",
        lr_warmup_step: int = 0,
        lr_num_cycles: int = 0.5,
        train_dataloader_len: int = None,
    ):
        super(CommonLitModel, self).__init__()
        self.save_hyperparameters()
        self.train_dataloader_len = train_dataloader_len

        self.roberta_model = CommonLitRoBERTaModel(
            model_name_or_path=roberta_model_name_or_path,
            output_hidden_states=output_hidden_states,
        )
        self.loss_fn = nn.MSELoss()
        self.eval_fn = RMSELoss()

    def forward(self, batch):
        z = self.roberta_model(batch)
        return z

    def configure_optimizers(self):
        optimizer_grouped_parameters = self._get_optimizer_params(self.roberta_model)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,  # self.parameters()
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )

        total_steps = self.train_dataloader_len * self.hparams.num_epochs
        if self.hparams.lr_scheduler == "linear":
            # Linear scheduler
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )
        elif self.hparams.lr_scheduler == "cosine":
            # Cosine scheduler
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.lr_warmup_step,
                num_training_steps=total_steps,
                num_cycles=self.hparams.lr_num_cycles,
            )
        else:
            # Linear scheduler
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.lr_warmup_step,
                num_training_steps=total_steps,
            )

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.lr_interval,  # step or epoch
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def _get_optimizer_params(self, model):
        # Differential learning rate and weight decay
        param_optimizer = list(model.named_parameters())
        learning_rate = self.hparams.lr
        no_decay = ["bias", "gamma", "beta"]
        group1 = ["layer.0.", "layer.1.", "layer.2.", "layer.3."]
        group2 = ["layer.4.", "layer.5.", "layer.6.", "layer.7."]
        group3 = ["layer.8.", "layer.9.", "layer.10.", "layer.11."]
        group_all = [
            "layer.0.",
            "layer.1.",
            "layer.2.",
            "layer.3.",
            "layer.4.",
            "layer.5.",
            "layer.6.",
            "layer.7.",
            "layer.8.",
            "layer.9.",
            "layer.10.",
            "layer.11.",
        ]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "weight_decay_rate": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group1)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate * 1e-1,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group2)
                ],
                "weight_decay_rate": 0.01,  #  0.0,
                "lr": learning_rate * 0.5,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group3)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "weight_decay_rate": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate * 1e-1,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate * 0.5,
            },
            {
                "params": [
                    p
                    for n, p in model.roberta.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "roberta" not in n
                ],
                "weight_decay_rate": 0.0,
                "lr": 2e-4,  # learning_rate
                "momentum": 0.99,
            },
        ]
        return optimizer_parameters

    def shared_step(self, batch):
        z = self(batch)
        return z

    def training_step(self, batch, batch_idx):
        z = self.shared_step(batch)
        loss = self.loss_fn(z, batch["target"])
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        z = self.shared_step(batch)
        return {"pred": z, "target": batch["target"]}

    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end(self, validation_step_outputs):
        pred = []
        target = []

        for output in validation_step_outputs:
            pred.append(output["pred"])
            target.append(output["target"])

        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)

        loss = self.loss_fn(pred, target)
        metric = self.eval_fn(pred, target)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)

    def get_roberta_state_dict(self):
        return self.roberta_model.state_dict()
