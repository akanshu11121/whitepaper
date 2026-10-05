"""Paper §5 optimizer/schedule/smoothing; local batching is documented in A10."""

from torch import Tensor
from torch.nn import functional as F

from research_lab.datasets import PAD


def noam_rate(step: int, d_model: int, warmup: int, factor: float = 1) -> float:
    """§5.3 Eq.3 with one-based optimizer steps; factor is an engineering control."""
    if step < 1 or warmup < 1:
        raise ValueError("step and warmup must be positive")
    return factor * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def smoothed_loss(logits: Tensor, targets: Tensor, epsilon: float) -> Tensor:
    """A3: exclude padding class and true class from smoothing support."""
    log_probs = F.log_softmax(logits, dim=-1)
    valid = targets.ne(PAD)
    if not valid.any():
        raise ValueError("loss requires at least one non-padding target")
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    alternative = (log_probs.sum(-1) - log_probs[..., PAD] - chosen) / (logits.shape[-1] - 2)
    return (-(1 - epsilon) * chosen - epsilon * alternative)[valid].mean()


def nll_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD)
