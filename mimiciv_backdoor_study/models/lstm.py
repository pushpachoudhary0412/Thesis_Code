"""
Simple LSTM model exposing .forward(x) -> logits

Accepts input of shape (batch, features) and treats features as a sequence of length `features`
by embedding each scalar with a linear layer and passing through an LSTM encoder, then pooling.
This keeps the interface consistent with other models in the scaffold.
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1, bidirectional: bool = False, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.emb = nn.Linear(1, emb_dim)  # embed each scalar feature to vector
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor (batch, features)
        returns: logits (batch, n_classes)
        """
        batch, feats = x.shape
        # treat features as sequence length = feats, with channel 1
        seq = x.view(batch, feats, 1)            # (B, L, 1)
        emb = self.emb(seq)                      # (B, L, emb_dim)
        out, _ = self.lstm(emb)                  # (B, L, hidden)
        # pool over sequence (L)
        out_t = out.transpose(1, 2)              # (B, hidden, L)
        pooled = self.pool(out_t).squeeze(-1)    # (B, hidden)
        logits = self.classifier(pooled)         # (B, n_classes)
        return logits
