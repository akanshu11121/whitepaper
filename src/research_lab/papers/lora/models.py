"""Small controlled models used to validate LoRA adaptation independently of an LLM."""

import torch
from torch import Tensor, nn

from research_lab.papers.lora.layers import LoRALinear


class AdaptationRegressor(nn.Module):
    def __init__(self, base_weight: Tensor, base_bias: Tensor | None, rank: int,
                 alpha: float, dropout: float, init_std: float):
        super().__init__()
        base = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=base_bias is not None)
        with torch.no_grad():
            base.weight.copy_(base_weight)
            if base_bias is not None and base.bias is not None:
                base.bias.copy_(base_bias)
        self.projection = LoRALinear.from_linear(base, rank=rank, alpha=alpha,
                                                 dropout=dropout, init_std=init_std)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)


class FullFineTuneRegressor(nn.Module):
    def __init__(self, base_weight: Tensor, base_bias: Tensor | None):
        super().__init__()
        self.projection = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=base_bias is not None)
        with torch.no_grad():
            self.projection.weight.copy_(base_weight)
            if base_bias is not None and self.projection.bias is not None:
                self.projection.bias.copy_(base_bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)
