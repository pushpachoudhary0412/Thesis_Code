"""
Simple Temporal CNN (TCN)-like model exposing .forward(x) -> logits

Treats features as a sequence and applies 1D dilated convolutions followed by pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    def __init__(self, input_dim: int, channels=(32, 64), kernel_size=3, dropout=0.1, n_classes=2):
        super().__init__()
        layers = []
        in_chan = 1  # input channel for scalar-per-time-step representation
        prev_channels = in_chan
        # initial projection to channels[0]
        self.input_proj = nn.Conv1d(prev_channels, channels[0], kernel_size=1)
        convs = []
        in_ch = channels[0]
        dilation = 1
        for ch in channels:
            convs.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=(kernel_size-1)//2 * dilation, dilation=dilation))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_ch = ch
            dilation *= 2
        self.tcn = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, features)
        Process as (batch, channels=1, length=features)
        returns logits (batch, n_classes)
        """
        b, feats = x.shape
        seq = x.view(b, 1, feats)         # (B, 1, L)
        h = self.input_proj(seq)          # (B, C0, L)
        h = self.tcn(h)                   # (B, C_last, L)
        pooled = self.pool(h).squeeze(-1) # (B, C_last)
        logits = self.classifier(pooled)  # (B, n_classes)
        return logits
