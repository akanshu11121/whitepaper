"""Controlled adaptation experiments, baselines, artifacts and benchmark data."""

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from research_lab.artifacts import ArtifactStore
from research_lab.papers.lora.config import LoraBenchmarkConfig, LoraExperimentConfig
from research_lab.papers.lora.layers import (
    LoRALinear,
    load_lora_state,
    mark_only_lora_trainable,
    merge_lora_modules,
    parameter_report,
    save_lora_state,
    unmerge_lora_modules,
)
from research_lab.papers.lora.models import AdaptationRegressor, FullFineTuneRegressor
from research_lab.runtime import environment, seed_all


@dataclass
class AdaptationData:
    base_weight: Tensor
    base_bias: Tensor
    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor
    target_delta: Tensor
    fingerprint: str


def make_data(config: LoraExperimentConfig) -> AdaptationData:
    seed_all(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    base_weight = torch.randn(config.output_dim, config.input_dim, generator=generator) / math.sqrt(config.input_dim)
    base_bias = torch.randn(config.output_dim, generator=generator) * 0.05
    true_a = torch.randn(config.target_rank, config.input_dim, generator=generator) / math.sqrt(config.input_dim)
    true_b = torch.randn(config.output_dim, config.target_rank, generator=generator) / math.sqrt(config.target_rank)
    target_delta = config.target_scale * config.effective_target_alpha / config.target_rank * (true_b @ true_a)
    total = config.train_size + config.eval_size
    inputs = torch.randn(total, config.input_dim, generator=generator)
    target_weight = base_weight + target_delta
    outputs = inputs @ target_weight.T + base_bias
    if config.noise_std:
        outputs = outputs + config.noise_std * torch.randn(outputs.shape, generator=generator)
    serialized = torch.cat([base_weight.flatten(), target_delta.flatten(), inputs.flatten(), outputs.flatten()]).numpy().tobytes()
    fingerprint = hashlib.sha256(serialized).hexdigest()
    return AdaptationData(base_weight, base_bias, inputs[:config.train_size], outputs[:config.train_size],
                          inputs[config.train_size:], outputs[config.train_size:], target_delta, fingerprint)


def mse(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(inputs) - targets) ** 2))


def train_model(model: nn.Module, inputs: Tensor, targets: Tensor, steps: int,
                batch_size: int, learning_rate: float, seed: int) -> list[dict[str, float]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return []
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(inputs), (min(batch_size, len(inputs)),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs[indices]) - targets[indices]) ** 2)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            history.append({"step": float(step), "loss": float(loss.detach())})
    return history


def _save_base(store: ArtifactStore, run_id: str, data: AdaptationData) -> None:
    save_file({"weight": data.base_weight.detach().cpu(), "bias": data.base_bias.detach().cpu()}, str(store.path(run_id) / "base.safetensors"))


def train(config: LoraExperimentConfig, store: ArtifactStore, run_id: str) -> dict[str, Any]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = config.device
    raw_data = make_data(config)
    data = AdaptationData(
        base_weight=raw_data.base_weight.to(device),
        base_bias=raw_data.base_bias.to(device),
        train_x=raw_data.train_x.to(device),
        train_y=raw_data.train_y.to(device),
        test_x=raw_data.test_x.to(device),
        test_y=raw_data.test_y.to(device),
        target_delta=raw_data.target_delta.to(device),
        fingerprint=raw_data.fingerprint,
    )
    lora = AdaptationRegressor(data.base_weight, data.base_bias, config.rank, config.effective_alpha, config.lora_dropout, config.init_std).to(device)
    full = FullFineTuneRegressor(data.base_weight, data.base_bias).to(device)
    frozen = AdaptationRegressor(data.base_weight, data.base_bias, 0, 1, 0, config.init_std).to(device)
    mark_only_lora_trainable(lora, config.bias_mode)
    mark_only_lora_trainable(frozen, "none")
    full_history = train_model(full, data.train_x, data.train_y, config.steps, config.batch_size, config.learning_rate, config.seed + 1)
    lora_history = train_model(lora, data.train_x, data.train_y, config.steps, config.batch_size, config.learning_rate, config.seed + 2)
    _save_base(store, run_id, data)
    save_lora_state(lora, str(store.path(run_id) / "adapter.safetensors"))
    lora_report = parameter_report(lora)
    full_report = parameter_report(full)
    frozen_report = parameter_report(frozen)
    merge_lora_modules(lora)
    merged_mse = mse(lora, data.test_x, data.test_y)
    unmerge_lora_modules(lora)
    result = {
        "status": "completed", "device": device, "scientific_status": "local-derived-result",
        "config": config.model_dump(), "data_fingerprint": data.fingerprint,
        "parameter_reports": {"lora": lora_report, "full_finetune": full_report, "frozen": frozen_report},
        "metrics": {"lora_test_mse": mse(lora, data.test_x, data.test_y), "merged_lora_test_mse": merged_mse,
                    "full_finetune_test_mse": mse(full, data.test_x, data.test_y), "frozen_test_mse": mse(frozen, data.test_x, data.test_y),
                    "target_delta_frobenius": float(torch.linalg.matrix_norm(data.target_delta))},
        "history": {"lora": lora_history, "full_finetune": full_history},
        "merge_max_difference": merge_difference(lora, data.test_x),
        "artifacts": {"base": "base.safetensors", "adapter": "adapter.safetensors"},
        "environment": environment(), "limitations": ["controlled regression is not an NLP downstream benchmark", "reported paper metrics are not reproduced"],
    }
    store.write(run_id, "dataset.json", {"fingerprint": data.fingerprint, "train_size": len(data.train_x), "test_size": len(data.test_x), "input_dim": config.input_dim, "output_dim": config.output_dim})
    store.update(run_id, **result)
    return result


def merge_difference(model: AdaptationRegressor, inputs: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        before = model(inputs)
        model.projection.merge()
        merged = model(inputs)
        model.projection.unmerge()
    return float((before - merged).abs().max())


def load_adapter(store: ArtifactStore, run_id: str) -> AdaptationRegressor:
    result = store.read(run_id)
    config = LoraExperimentConfig.model_validate(result["config"])
    base = load_file(str(store.path(run_id) / "base.safetensors"))
    model = AdaptationRegressor(base["weight"], base["bias"], config.rank, config.effective_alpha, config.lora_dropout, config.init_std)
    load_lora_state(model, str(store.path(run_id) / "adapter.safetensors"))
    model.eval()
    return model


def predict(store: ArtifactStore, run_id: str, values: str) -> dict[str, Any]:
    result = store.read(run_id)
    config = LoraExperimentConfig.model_validate(result["config"])
    numbers = [float(value) for value in values.replace(",", " ").split()]
    if len(numbers) != config.input_dim:
        raise ValueError(f"expected exactly {config.input_dim} numeric input values")
    model = load_adapter(store, run_id)
    vector = torch.tensor([numbers], dtype=model.projection.weight.dtype)
    with torch.no_grad():
        unmerged = model(vector).squeeze(0).tolist()
        model.projection.merge()
        merged = model(vector).squeeze(0).tolist()
        model.projection.unmerge()
    return {"input": numbers, "output": unmerged, "merged_output": merged,
            "max_merge_difference": max(abs(a - b) for a, b in zip(unmerged, merged, strict=True)),
            "note": "Regression-vector output; this module does not tokenize or generate language."}


def benchmark(config: LoraBenchmarkConfig) -> dict[str, Any]:
    seed_all(config.seed)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    base = nn.Linear(config.input_dim, config.output_dim, bias=True, dtype=dtype)
    model = LoRALinear.from_linear(base, config.rank, config.alpha, init_std=0.02).to(dtype=dtype).eval()
    inputs = torch.randn(config.batch_size, config.input_dim, dtype=dtype)
    for _ in range(config.warmup):
        model(inputs)
    unmerged = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        model(inputs)
        unmerged.append((time.perf_counter() - start) * 1000)
    model.merge()
    merged = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        model(inputs)
        merged.append((time.perf_counter() - start) * 1000)
    model.unmerge()
    def stats(values: list[float]) -> dict[str, float]:
        return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values))}
    report = parameter_report(model, 8 if dtype == torch.float64 else 4)
    return {"status": "measured", "scientific_status": "engineering-measurement", "config": config.model_dump(),
            "parameter_report": report, "unmerged": stats(unmerged), "merged": stats(merged),
            "speedup_merged_over_unmerged": float(np.mean(unmerged) / max(np.mean(merged), 1e-12)),
            "environment": environment(), "note": "CPU timing includes Python/framework overhead; no GPU or production throughput claim."}
