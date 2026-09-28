"""Loss function registry, so the loss can be chosen (and tuned) by name.

Classification losses expect raw logits of shape (batch, n_classes) and
integer targets; regression losses expect (batch,) predictions and targets.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): down-weights easy examples so training
    focuses on hard / minority-class ones.  ``gamma=0`` is plain cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


CLASSIFICATION_LOSSES = ["cross_entropy", "weighted_cross_entropy", "label_smoothing", "focal"]
REGRESSION_LOSSES = ["mse", "mae", "huber", "smooth_l1"]


def get_loss(
    name: str,
    class_weights: Optional[Sequence[float]] = None,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    huber_delta: float = 1.0,
) -> nn.Module:
    """Build a loss by name.

    Classification: ``cross_entropy``, ``weighted_cross_entropy`` (needs
    ``class_weights``), ``label_smoothing``, ``focal``.
    Regression: ``mse``, ``mae``, ``huber``, ``smooth_l1``.
    """
    weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_cross_entropy":
        if weight is None:
            raise ValueError("weighted_cross_entropy needs class_weights")
        return nn.CrossEntropyLoss(weight=weight)
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=weight)

    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=huber_delta)

    raise ValueError(f"Unknown loss '{name}'")
