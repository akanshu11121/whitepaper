"""Paper §3: post-norm encoder–decoder Transformer, not a modern pre-norm variant."""

import math

import torch
from torch import Tensor, nn

from research_lab.config import ModelConfig
from research_lab.datasets import PAD
from research_lab.papers.attention.attention import MultiHeadAttention


def sinusoidal_positions(length: int, d_model: int) -> Tensor:
    """§3.5; also supports odd widths as an engineering convenience."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
    result = torch.zeros(length, d_model)
    result[:, 0::2] = torch.sin(position * frequencies)
    result[:, 1::2] = torch.cos(position * frequencies[:d_model // 2])
    return result


def feed_forward(config: ModelConfig) -> nn.Sequential:
    """§3.3 Eq.2, same transformation independently at every token position."""
    return nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.ReLU(),
                         nn.Linear(config.d_ff, config.d_model))


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config.d_model, config.heads, config.attention_dropout, config.optimized)
        self.ff = feed_forward(config)
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(2)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, capture: bool) -> tuple[Tensor, Tensor | None]:
        attended, weights = self.attention(x, x, x, mask, capture)
        x = self.norms[0](x + self.dropout(attended))
        return self.norms[1](x + self.dropout(self.ff(x))), weights


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.heads, config.attention_dropout, config.optimized)
        self.cross_attention = MultiHeadAttention(config.d_model, config.heads, config.attention_dropout, config.optimized)
        self.ff = feed_forward(config)
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(3)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, memory: Tensor, source_mask: Tensor,
                target_mask: Tensor, capture: bool) -> tuple[Tensor, Tensor | None, Tensor | None]:
        attended, self_weights = self.self_attention(x, x, x, target_mask, capture)
        x = self.norms[0](x + self.dropout(attended))
        attended, cross_weights = self.cross_attention(x, memory, memory, source_mask, capture)
        x = self.norms[1](x + self.dropout(attended))
        return self.norms[2](x + self.dropout(self.ff(x))), self_weights, cross_weights


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        # §3.4: the same embedding is used for source, target, and pre-softmax weights.
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("positions", sinusoidal_positions(config.max_length, config.d_model), persistent=False)
        self.learned_positions = nn.Embedding(config.max_length, config.d_model) if config.positions == "learned" else None
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.layers)])
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.embedding.weight, std=config.d_model**-0.5)

    def embed(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds model positional capacity")
        x = self.embedding(tokens) * math.sqrt(self.config.d_model)
        if self.config.positions == "sinusoidal":
            positions = self.positions
            if not isinstance(positions, Tensor):
                raise RuntimeError("positional buffer is not a tensor")
            x = x + positions[:tokens.shape[1]]
        elif self.learned_positions is not None:
            x = x + self.learned_positions(torch.arange(tokens.shape[1], device=tokens.device))
        return self.dropout(x)

    def encode(self, source: Tensor, capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        x = self.embed(source)
        trace: dict[str, list[Tensor]] = {"encoder": [], "encoder_states": []}
        mask = source.ne(PAD)[:, None, None, :]
        for layer in self.encoder:
            x, weights = layer(x, mask, capture)
            if capture and weights is not None:
                trace["encoder"].append(weights)
                trace["encoder_states"].append(x)
        return x, trace

    def decode(self, target: Tensor, memory: Tensor, source: Tensor,
               capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        x = self.embed(target)
        length = target.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=target.device).tril()
        target_mask = target.ne(PAD)[:, None, None, :] & causal[None, None, :, :]
        source_mask = source.ne(PAD)[:, None, None, :]
        trace: dict[str, list[Tensor]] = {"decoder": [], "cross": [], "decoder_states": []}
        for layer in self.decoder:
            x, self_weights, cross_weights = layer(x, memory, source_mask, target_mask, capture)
            if capture and self_weights is not None and cross_weights is not None:
                trace["decoder"].append(self_weights)
                trace["cross"].append(cross_weights)
                trace["decoder_states"].append(x)
        return self.output(x), trace

    def forward(self, source: Tensor, target: Tensor,
                capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]:
        memory, encoder_trace = self.encode(source, capture)
        logits, decoder_trace = self.decode(target, memory, source, capture)
        return logits, {**encoder_trace, **decoder_trace}
