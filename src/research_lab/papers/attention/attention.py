"""Paper §3.2: explicit Eq.1 reference path and equivalent PyTorch SDPA path."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def scaled_attention(q: Tensor, k: Tensor, v: Tensor, allowed: Tensor | None = None,
                     dropout: float = 0, training: bool = False) -> tuple[Tensor, Tensor]:
    """Eq.1. Boolean mask True means allowed; all-masked rows yield zero (A7)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise ValueError("attention mask must be boolean (True=allowed)")
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(allowed.any(dim=-1, keepdim=True), scores, 0)
    weights = scores.softmax(dim=-1)
    if allowed is not None:
        weights = weights.masked_fill(~allowed, 0)
    return F.dropout(weights, p=dropout, training=training) @ v, weights


class MultiHeadAttention(nn.Module):
    """§3.2.2 independent learned projections and head concatenation; no biases."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0, optimized: bool = False):
        super().__init__()
        if d_model % heads:
            raise ValueError("model width must divide evenly into heads")
        self.heads, self.head_dim = heads, d_model // heads
        self.dropout, self.optimized = dropout, optimized
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                allowed: Tensor | None = None, capture: bool = False) -> tuple[Tensor, Tensor | None]:
        def split(x: Tensor, projection: nn.Linear) -> Tensor:
            return projection(x).view(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(query, self.q), split(key, self.k), split(value, self.v)
        weights = None
        if self.optimized and not capture:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                   dropout_p=self.dropout if self.training else 0)
        else:
            output, weights = scaled_attention(q, k, v, allowed, self.dropout, self.training)
        combined = output.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
        return self.out(combined), weights if capture else None
