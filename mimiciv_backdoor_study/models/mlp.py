"""
Simple MLP model exposing .forward(x) -> logits

Designed for tabular features from the synthetic dev dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), n_classes=2, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, features)
        returns: logits Tensor of shape (batch, n_classes)
        """
        return self.net(x)
