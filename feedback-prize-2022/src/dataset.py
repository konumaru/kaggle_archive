import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class FeedbackDataset(Dataset):
    def __init__(self, inputs, tokenizer, targets=None, max_seq_len: int = 512):
        super().__init__()
        self.inputs = inputs

        if targets is None:
            self.targets = np.empty((len(inputs), 6))
        else:
            self.targets = targets.to_numpy()

        self.encoded = tokenizer(
            inputs["full_text"].tolist(),
            padding="max_length",
            add_special_tokens=True,
            max_length=max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = torch.from_numpy(self.encoded["input_ids"][index]).to(torch.int32)
        mask = torch.from_numpy(self.encoded["attention_mask"][index]).to(torch.int32)
        labels = torch.from_numpy(self.targets[index]).to(torch.float32)

        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


def get_train_dataloader(
    inputs, tokenizer, batch_size: int = 8, max_seq_len: int = 512
):
    dataset = FeedbackDataset(
        inputs[["text_id", "full_text"]],
        tokenizer,
        inputs[
            [
                "cohesion",
                "syntax",
                "vocabulary",
                "phraseology",
                "grammar",
                "conventions",
            ]
        ],
        max_seq_len=max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )
    return dataloader


def get_valid_dataloader(
    inputs, tokenizer, batch_size: int = 8, max_seq_len: int = 512
):
    dataset = FeedbackDataset(
        inputs[["text_id", "full_text"]],
        tokenizer,
        inputs[
            [
                "cohesion",
                "syntax",
                "vocabulary",
                "phraseology",
                "grammar",
                "conventions",
            ]
        ],
        max_seq_len=max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )
    return dataloader


def get_inference_dataloader(
    inputs, tokenizer, batch_size: int = 8, max_seq_len: int = 512
):
    dataset = FeedbackDataset(
        inputs[["text_id", "full_text"]],
        tokenizer,
        max_seq_len=max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return dataloader


def main():
    data = pd.read_csv("../data/raw/train.csv").head(32)
    print(data.head())

    train, test = model_selection.train_test_split(data)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/mdeberta-v3-base", use_fast=True
    )

    train_dataloader = get_train_dataloader(train, tokenizer, 8)
    batch = next(iter(train_dataloader))
    print(batch)

    test_dataloader = get_train_dataloader(test, tokenizer, 8)
    batch = next(iter(test_dataloader))
    print(batch)


if __name__ == "__main__":
    main()
