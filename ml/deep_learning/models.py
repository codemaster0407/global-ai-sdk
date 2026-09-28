"""Neural network architectures for tabular data."""
from typing import Sequence

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class TabularMLP(nn.Module):
    """Feed-forward network: [Linear -> BatchNorm -> Activation -> Dropout] x N -> Linear.

    ``out_features`` is n_classes for classification (raw logits) or 1 for regression.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        prev = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
