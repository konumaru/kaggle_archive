from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from .dataset import CommonLitDataset
from .metrics import MCRMSELoss


class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CommonLitModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.007,
            }
        )
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooler = MeanPooling()

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.num_labels),
        )

        self.init_model()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.pooler(output.last_hidden_state, attention_mask)
        output = self.head(output)
        return output

    def init_model(self) -> None:
        for i in range(0, 6):
            for _, param in self.model.encoder.layer[i].named_parameters():
                param.requires_grad = False

        for layer in self.model.encoder.layer[-4:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=self.model.config.initializer_range
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(
                        mean=0.0, std=self.model.config.initializer_range
                    )
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
