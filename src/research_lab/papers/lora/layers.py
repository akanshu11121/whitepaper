"""Paper §4.1 LoRA layer, explicit merge semantics, and adapter accounting."""

from collections.abc import Iterable
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """A paper-faithful frozen linear layer plus a trainable low-rank update.

    The base weight is stored in the inherited ``weight`` parameter and frozen.
    ``lora_A`` has shape [rank, in_features], ``lora_B`` [out_features, rank],
    matching W = W0 + (alpha/rank) B A for PyTorch's y = x Wᵀ convention.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 0,
                 alpha: float = 1.0, dropout: float = 0.0, bias: bool = True,
                 init_std: float = 0.02):
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if rank > min(in_features, out_features):
            raise ValueError("rank must not exceed min(in_features, out_features)")
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank if rank else 0.0
        self.init_std = init_std
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.merged = False
        if rank:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.reset_parameters()
        self.weight.requires_grad_(rank == 0)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = fan_in**-0.5
            nn.init.uniform_(self.bias, -bound, bound)
        rank = getattr(self, "rank", 0)
        if rank and hasattr(self, "lora_A") and self.lora_A is not None and self.lora_B is not None:
            # Paper §4.1: random Gaussian A, zero B, so Delta W begins at zero.
            nn.init.normal_(self.lora_A, mean=0.0, std=self.init_std)
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int, alpha: float | None = None,
                    dropout: float = 0.0, init_std: float = 0.02) -> "LoRALinear":
        if layer.weight.ndim != 2:
            raise ValueError("LoRALinear requires a two-dimensional linear weight")
        result = cls(layer.in_features, layer.out_features, rank,
                     alpha=float(rank if alpha is None else alpha), dropout=dropout,
                     bias=layer.bias is not None, init_std=init_std)
        with torch.no_grad():
            result.weight.copy_(layer.weight)
            if layer.bias is not None and result.bias is not None:
                result.bias.copy_(layer.bias)
        result.weight.requires_grad_(False)
        if result.bias is not None:
            result.bias.requires_grad_(False)
        return result

    @property
    def delta_weight(self) -> Tensor:
        if not self.rank or self.lora_A is None or self.lora_B is None:
            return torch.zeros_like(self.weight)
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, input: Tensor) -> Tensor:
        base = F.linear(input, self.weight, self.bias)
        if self.rank == 0 or self.merged:
            return base
        update = self.lora_dropout(input) @ self.lora_A.transpose(0, 1)
        update = update @ self.lora_B.transpose(0, 1)
        return base + self.scaling * update

    def merge(self) -> None:
        if self.rank and not self.merged:
            with torch.no_grad():
                self.weight.add_(self.delta_weight.to(device=self.weight.device, dtype=self.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        if self.rank and self.merged:
            with torch.no_grad():
                self.weight.sub_(self.delta_weight.to(device=self.weight.device, dtype=self.weight.dtype))
            self.merged = False

    def adapter_state(self) -> dict[str, Tensor]:
        if not self.rank or self.lora_A is None or self.lora_B is None:
            return {}
        return {"lora_A": self.lora_A.detach().cpu(), "lora_B": self.lora_B.detach().cpu()}


def iter_lora_modules(module: nn.Module) -> Iterable[LoRALinear]:
    return (child for child in module.modules() if isinstance(child, LoRALinear))


def merge_lora_modules(module: nn.Module) -> None:
    for child in iter_lora_modules(module):
        child.merge()


def unmerge_lora_modules(module: nn.Module) -> None:
    for child in iter_lora_modules(module):
        child.unmerge()


def mark_only_lora_trainable(module: nn.Module, bias: str = "none") -> None:
    if bias not in {"none", "all"}:
        raise ValueError("bias must be none or all")
    for name, parameter in module.named_parameters():
        parameter.requires_grad_("lora_" in name or (bias == "all" and name.endswith("bias")))


def parameter_report(module: nn.Module, dtype_bytes: int = 4) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    adapter = sum(parameter.numel() for name, parameter in module.named_parameters() if "lora_" in name)
    return {
        "total_parameters": total, "trainable_parameters": trainable,
        "adapter_parameters": adapter, "frozen_parameters": total - trainable,
        "trainable_fraction": trainable / max(total, 1),
        "full_model_bytes": total * dtype_bytes, "trainable_bytes": trainable * dtype_bytes,
        "adapter_bytes": adapter * dtype_bytes,
    }


def lora_state_dict(module: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for prefix, child in module.named_modules():
        if isinstance(child, LoRALinear):
            for name, value in child.adapter_state().items():
                state[f"{prefix}.{name}" if prefix else name] = value
    return state


def save_lora_state(module: nn.Module, path: str) -> None:
    state = lora_state_dict(module)
    if not state:
        raise ValueError("module has no LoRA parameters")
    save_file(state, path)


def load_lora_state(module: nn.Module, path: str) -> None:
    state = load_file(path)
    missing, unexpected = module.load_state_dict(state, strict=False)
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    missing_lora = [key for key in missing if "lora_" in key]
    if unexpected_lora or missing_lora:
        raise ValueError(f"adapter state mismatch: missing={missing_lora}, unexpected={unexpected_lora}")
