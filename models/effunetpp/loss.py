import torch
import torch.nn as nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.ignore_label = -1
        self.smooth = 0
        self.exponent = 2
        weight = torch.tensor([0.5, 2.5])
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
        self.dice = 0
        self.acc = 0

    def binary_dice_loss(self, pred, target, valid_mask):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(
            pred.pow(self.exponent) * valid_mask
            + target.pow(self.exponent) * valid_mask,
            dim=1,
        ) + max(self.smooth, 1e-5)

        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice

    def forward(self, score, pred_label, target, labels):
        target = target.squeeze(1).long()
        labels = labels.unsqueeze(1).float()
        CE_loss = self.cross_entropy(score, target)
        score = F.softmax(score, dim=1)
        num_classes = score.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )
        valid_mask = (target != self.ignore_label).long()
        dice_loss = self.binary_dice_loss(
            score[:, 1], one_hot_target[..., 1], valid_mask
        )
        self.dice = 1 - dice_loss
        BCE_loss = nn.BCEWithLogitsLoss()(pred_label, labels)
        # print(pred_label, labels)
        # print(((pred_label > 0).float() == labels).float());exit()
        self.acc = ((pred_label > 0).float() == labels).float().mean()

        return 0.2 * CE_loss + 0.7 * dice_loss + 0.1 * BCE_loss
