"""Paper §6.2-inspired local ablations; each variant is a separate persisted run."""

from typing import Any

from research_lab.artifacts import ArtifactStore
from research_lab.config import ExperimentConfig
from research_lab.datasets import build_dataset
from research_lab.papers.attention.runner import train


def presets(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    variants: dict[str, ExperimentConfig] = {}
    changes_by_name: dict[str, dict[str, Any]] = {
        "full": {}, "single_head": {"model": {"heads": 1}},
        "no_smoothing": {"smoothing": 0}, "learned_positions": {"model": {"positions": "learned"}},
        "no_positions": {"model": {"positions": "none"}}, "smaller_ffn": {"model": {"d_ff": max(8, base.model.d_model)}}
    }
    for name, changes in changes_by_name.items():
        payload: dict[str, Any] = base.model_dump()
        payload.update({key: value for key, value in changes.items() if key != "model"})
        if "model" in changes:
            payload["model"] = {**payload["model"], **changes["model"]}
        variants[name] = ExperimentConfig.model_validate(payload)
    return variants


def run_ablations(base: ExperimentConfig, store: ArtifactStore) -> dict[str, Any]:
    results = {}
    for name, config in presets(base).items():
        run_id = store.create("attention", f"ablation:{name}", config.model_dump())
        result = train(config, build_dataset(config), store, run_id)
        results[name] = {"run_id": run_id, "test": result["test"], "duration_seconds": result["duration_seconds"], "parameter_count": result["parameter_count"]}
    return results
