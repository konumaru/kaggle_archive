import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        bce = self.bce(input, target)
        dice = self._dice_loss(input, target)
        return bce + dice

    def _dice_loss(self, inputs, targets, eps=1e-7):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum(dim=1) + eps
        cardinality = inputs.sum(dim=1) + targets.sum(dim=1) + eps
        dice = ((2.0 * intersection) / (intersection + cardinality)).mean()
        return 1 - dice
