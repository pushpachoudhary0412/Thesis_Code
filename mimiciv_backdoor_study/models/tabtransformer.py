"""
Lightweight TabTransformer-like model for tabular sequences.

This is a simplified implementation intended for experiments on the synthetic dev dataset.
For production research, consider using a full TabTransformer implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTabTransformer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, n_heads: int = 4, n_layers: int = 2, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        # project each scalar feature to embed_dim
        self.proj = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, features)
        returns logits (batch, n_classes)
        """
        b, feats = x.shape
        seq = x.view(b, feats, 1)          # (B, L, 1)
        emb = self.proj(seq)               # (B, L, embed_dim)
        h = self.transformer(emb)          # (B, L, embed_dim)
        h_t = h.transpose(1, 2)            # (B, embed_dim, L)
        pooled = self.pool(h_t).squeeze(-1) # (B, embed_dim)
        logits = self.classifier(pooled)   # (B, n_classes)
        return logits
