from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str,
        targets: List[str] = ["content", "wording"],
        max_len: int = 512,
        is_train: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.input_texts = (
            data["prompt_question"]
            + f" {self.tokenizer.sep_token} "
            + data["text"]
        ).tolist()
        self.prompt_q = data["prompt_question"].tolist()
        self.max_len = max_len

        if is_train:
            self.targets = torch.from_numpy(data[targets].to_numpy())
        else:
            self.targets = torch.zeros((len(data), len(targets)))

    def __len__(self) -> int:
        return len(self.input_texts)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        targets = self.targets[index, :]

        inputs = {}
        text = self.input_texts[index]
        encoded_token = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        inputs["input_ids"] = encoded_token[
            "input_ids"
        ].squeeze()  # type: ignore
        inputs["attention_mask"] = encoded_token[
            "attention_mask"
        ].squeeze()  # type: ignore

        return (inputs, targets)
