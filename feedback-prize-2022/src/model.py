from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dataset import get_train_dataloader


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, hidden_state, mask):
        bmask = mask.unsqueeze(dim=-1).expand(hidden_state.shape)
        bmask_sum = torch.clamp(bmask.sum(dim=1), min=1e-9, max=None)
        embedding_sum = torch.sum(hidden_state * bmask, dim=1)
        mean_embeddings = embedding_sum / bmask_sum
        return mean_embeddings


class FeedbackModel(torch.nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.base_model = base_model

        self.config = self._get_config()
        self.transformer_model = AutoModel.from_pretrained(
            base_model, config=self.config
        )
        self._init_layers(num_init_layers=1)

        self.pool = MeanPooling()
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("normalize", nn.LayerNorm(self.config.hidden_size)),
                    ("output", nn.Linear(self.config.hidden_size, 6, bias=False)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )
        self._init_weight(self.head)

    def forward(self, inputs, mask):
        outputs = self.transformer_model(inputs, attention_mask=mask)
        feature = self.pool(outputs.last_hidden_state, mask)
        x = self.head(feature)
        x = x * 4.0 + 1.0
        return x, feature

    def _get_config(self):
        config = AutoConfig.from_pretrained(self.base_model)
        config.hidden_dropout = 0.0
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        config.output_hidden_states = True
        return config

    def _init_layers(self, num_init_layers=4):
        for layer in self.transformer_model.encoder.layer[-num_init_layers:]:
            for module in layer.modules():
                self._init_weight(module)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-9

    def forward(self, y_true, y_pred):
        error = torch.pow(torch.abs(y_true - y_pred), 2)
        mse = torch.mean(error, dim=0) + self.eps
        rmse = torch.sqrt(mse)
        return rmse.mean()


def get_optimizer_params(model: nn.Module):
    model_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    low_lr_params = [f"encoder.layer.{i}" for i in range(8)]
    high_lr_params = [f"encoder.layer.{i}" for i in range(8, 12)]

    optimizer_parameters = []
    for name, params in model_parameters:
        if not any(nd in name for nd in no_decay):
            if any([p in name for p in low_lr_params]):
                optimizer_parameters.append(
                    {"params": params, "weight_decay": 1e-2, "lr": 1e-5}
                )
            elif any([p in name for p in high_lr_params]):
                optimizer_parameters.append(
                    {"params": params, "weight_decay": 1e-2, "lr": 1e-4}
                )
            elif any([p in name for p in ["output"]]):
                optimizer_parameters.append(
                    {"params": params, "weight_decay": 1e-2, "lr": 1e-4}
                )

    return optimizer_parameters


def main():
    model_file = "microsoft/deberta-v3-base"
    data = pd.read_csv("../data/raw/train.csv").head(8)
    tokenizer = AutoTokenizer.from_pretrained(model_file, use_fast=True)
    dataloader = get_train_dataloader(data, tokenizer, 8, 512)
    batch = next(iter(dataloader))

    model = FeedbackModel(model_file)
    output, _ = model(batch["input_ids"], batch["attention_mask"])
    print(output)

    criterion = nn.SmoothL1Loss(reduction="mean")
    loss = criterion(batch["labels"], output)
    loss.backward()
    print(loss.item())

    optimizer_parameters = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-5, weight_decay=3e-1)


if __name__ == "__main__":
    main()
