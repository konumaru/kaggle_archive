import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MCRMSELoss, self).__init__()

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss
