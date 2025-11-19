"""
Lightweight TabTransformer-like model for tabular sequences.

This is a simplified implementation intended for experiments on the synthetic dev dataset.
For production research, consider using a full TabTransformer implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomMultiHeadAttention(nn.Module):
    """Custom MultiHeadAttention that saves attention weights."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights = None

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, embed_dim = query.shape

        # Linear projections
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Save attention weights for analysis
        self.attention_weights = attn_weights.detach()

        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class SimpleTabTransformer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, n_heads: int = 4, n_layers: int = 2, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        # project each scalar feature to embed_dim
        self.proj = nn.Linear(1, embed_dim)

        # Custom transformer layers for attention extraction
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
        )

        # For attention extraction
        self.extract_attention = False
        self.saved_attention = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, features)
        returns logits (batch, n_classes)
        """
        b, feats = x.shape
        seq = x.view(b, feats, 1)          # (B, L, 1)
        emb = self.proj(seq)               # (B, L, embed_dim)

        # Store attention weights if requested
        if self.extract_attention:
            self.saved_attention = []

        # Apply transformer layers
        h = emb
        for layer in self.layers:
            h = layer(h)
            if self.extract_attention:
                # Save attention from this layer (n_samples, n_heads, seq_len, seq_len)
                attn = layer.self_attn.attention_weights
                if attn is not None:
                    self.saved_attention.append(attn.cpu())

        h_t = h.transpose(1, 2)            # (B, embed_dim, L)
        pooled = self.pool(h_t).squeeze(-1) # (B, embed_dim)
        logits = self.classifier(pooled)   # (B, n_classes)
        return logits

    def get_attention_weights(self):
        """Return saved attention weights from last forward pass."""
        return self.saved_attention
