"""Autoregressive greedy/beam decoding (§6.1) and compatible checkpoint averaging."""

from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from research_lab.datasets import BOS, EOS, PAD
from research_lab.papers.attention.baseline import GRUSeq2Seq
from research_lab.papers.attention.model import Transformer

SequenceModel = Transformer | GRUSeq2Seq


class GenerationModel(Protocol):
    def eval(self) -> Any: ...
    def __call__(self, source: Tensor, target: Tensor,
                 capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]: ...
    def encode(self, source: Tensor, capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]: ...
    def decode(self, target: Tensor, memory: Tensor, source: Tensor,
               capture: bool = False) -> tuple[Tensor, dict[str, list[Tensor]]]: ...


@torch.inference_mode()
def generate(model: GenerationModel, source: Tensor, max_tokens: int,
             beam_size: int = 1, alpha: float = 0.6) -> tuple[list[list[int]], list[list[float]]]:
    """Recomputes target prefixes; no KV cache. Beam search runs per input example."""
    model.eval()
    if beam_size == 1:
        memory, _ = model.encode(source)
        target = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
        confidences: list[list[float]] = [[] for _ in range(source.shape[0])]
        for _ in range(max_tokens):
            logits, _ = model.decode(target, memory, source)
            scores = logits[:, -1].clone()
            scores[:, [PAD, BOS]] = -torch.inf
            probabilities = scores.softmax(-1)
            value, next_token = probabilities.max(-1)
            for i in range(source.shape[0]):
                if not finished[i]:
                    confidences[i].append(float(value[i]))
            next_token = torch.where(finished, EOS, next_token)
            target = torch.cat([target, next_token[:, None]], dim=1)
            finished |= next_token.eq(EOS)
            if finished.all():
                break
        return target.tolist(), confidences

    all_tokens, all_confidence = [], []
    for row in source:
        src = row.unsqueeze(0)
        memory, _ = model.encode(src)
        beams: list[tuple[list[int], float, list[float]]] = [([BOS], 0, [])]

        def normalized(beam: tuple[list[int], float, list[float]]) -> float:
            return beam[1] / (((5 + len(beam[0]) - 1) / 6) ** alpha)

        for _ in range(max_tokens):
            candidates = []
            for tokens, score, confidence in beams:
                if tokens[-1] == EOS:
                    candidates.append((tokens, score, confidence))
                    continue
                logits, _ = model.decode(torch.tensor([tokens], device=source.device), memory, src)
                scores = logits[0, -1].clone()
                scores[[PAD, BOS]] = -torch.inf
                log_probs = scores.log_softmax(-1)
                values, indices = log_probs.topk(min(beam_size, scores.shape[0] - 2))
                for value, index in zip(values.tolist(), indices.tolist(), strict=True):
                    candidates.append((tokens + [index], score + value, confidence + [float(torch.exp(torch.tensor(value)))]))
            beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
            if all(tokens[-1] == EOS for tokens, _, _ in beams):
                break
        best = max(beams, key=normalized)
        all_tokens.append(best[0])
        all_confidence.append(best[2])
    return all_tokens, all_confidence


def average_checkpoints(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("at least one checkpoint is required")
    states = [load_file(str(path)) for path in paths]
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("checkpoint keys differ")
    averaged = {}
    for key in keys:
        values = [state[key] for state in states]
        if any(value.shape != values[0].shape for value in values):
            raise ValueError("checkpoint shapes differ")
        averaged[key] = torch.stack(values).mean(0)
    save_file(averaged, str(output))
