import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from research_lab.artifacts import ArtifactStore
from research_lab.papers.lora.config import LoraBenchmarkConfig, LoraExperimentConfig
from research_lab.papers.lora.layers import (
    LoRALinear,
    load_lora_state,
    lora_state_dict,
    mark_only_lora_trainable,
    parameter_report,
    save_lora_state,
)
from research_lab.papers.lora.runner import benchmark, train


def test_forward_equation_and_zero_initial_update():
    torch.manual_seed(3)
    layer = LoRALinear(4, 3, rank=2, alpha=4)
    x = torch.randn(5, 4)
    base = nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(layer(x), base)
    assert torch.all(layer.delta_weight == 0)
    assert layer.lora_A.shape == (2, 4)
    assert layer.lora_B.shape == (3, 2)


def test_gradient_isolation_and_parameter_accounting():
    layer = LoRALinear(4, 3, rank=2, alpha=2)
    mark_only_lora_trainable(layer)
    output = layer(torch.ones(2, 4)).sum()
    output.backward()
    assert layer.weight.grad is None
    assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
    report = parameter_report(layer)
    assert report["adapter_parameters"] == 2 * 4 + 3 * 2
    assert report["trainable_parameters"] == report["adapter_parameters"]


def test_merge_and_unmerge_are_equivalent_and_idempotent():
    torch.manual_seed(4)
    layer = LoRALinear(4, 3, rank=2, alpha=2)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(2, 4)
    expected = layer(x)
    layer.merge()
    merged = layer(x)
    layer.merge()
    twice = layer(x)
    layer.unmerge()
    restored = layer(x)
    assert torch.allclose(expected, merged, atol=1e-6)
    assert torch.allclose(merged, twice, atol=1e-6)
    assert torch.allclose(expected, restored, atol=1e-6)


def test_adapter_state_round_trip():
    first = LoRALinear(4, 3, rank=2, alpha=2)
    with torch.no_grad():
        first.lora_B.normal_()
    second = LoRALinear.from_linear(nn.Linear(4, 3), rank=2, alpha=2)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "adapter.safetensors"
        save_lora_state(first, str(path))
        load_lora_state(second, str(path))
    assert set(lora_state_dict(first)) == {"lora_A", "lora_B"}
    assert torch.allclose(first.lora_A, second.lora_A)
    assert torch.allclose(first.lora_B, second.lora_B)


def test_invalid_rank_is_rejected():
    with pytest.raises(ValueError):
        LoRALinear(2, 3, rank=3)
    with pytest.raises(ValueError):
        LoraExperimentConfig(input_dim=2, output_dim=3, rank=3)


def test_controlled_training_persists_lora_artifact_and_beats_frozen_baseline():
    config = LoraExperimentConfig(seed=5, input_dim=8, output_dim=5, rank=2, target_rank=2,
                                  steps=35, train_size=64, eval_size=16, batch_size=16,
                                  learning_rate=0.08)
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(directory)
        run_id = store.create("lora", "experiment", config.model_dump())
        result = train(config, store, run_id)
        assert result["status"] == "completed"
        assert result["metrics"]["lora_test_mse"] < result["metrics"]["frozen_test_mse"]
        assert result["merge_max_difference"] < 1e-5
        assert (store.path(run_id) / "adapter.safetensors").exists()


def test_lora_benchmark_reports_merged_and_unmerged_paths():
    result = benchmark(LoraBenchmarkConfig(input_dim=16, output_dim=16, rank=2, repeats=5, warmup=2))
    assert result["unmerged"]["p50_ms"] > 0
    assert result["merged"]["p50_ms"] > 0
    assert result["parameter_report"]["adapter_parameters"] == 64
