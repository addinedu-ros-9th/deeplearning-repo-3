import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        TP = (preds * targets_onehot).sum(dim=(0, 2, 3))
        FP = (preds * (1 - targets_onehot)).sum(dim=(0, 2, 3))
        FN = ((1 - preds) * targets_onehot).sum(dim=(0, 2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow((1 - tversky), self.gamma)
        return focal_tversky.mean()
