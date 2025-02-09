import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import CommonLitModel
from models.dataset import CommonLitDataset
from models.metrics import MCRMSELoss


def tests_model() -> None:
    data = pd.read_csv("./data/preprocessed/train.csv")

    model_name = "microsoft/deberta-v3-base"
    num_labels = 2
    batch_size = 2

    model = CommonLitModel(model_name, num_labels=num_labels)
    assert isinstance(model, nn.Module)

    dataset = CommonLitDataset(data, model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    inputs, targets = next(iter(dataloader))
    z = model(inputs)
    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, num_labels)

    loss_fn = MCRMSELoss()
    loss = loss_fn(z, targets)
    loss.backward()
    assert isinstance(loss, torch.Tensor)
