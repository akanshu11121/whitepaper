"""LoRA rank/alpha ablation runner using identical generated data settings."""

from typing import Any

from research_lab.artifacts import ArtifactStore
from research_lab.papers.lora.config import LoraExperimentConfig
from research_lab.papers.lora.runner import train


def run_ablations(base: LoraExperimentConfig, store: ArtifactStore) -> dict[str, Any]:
    variants = {
        "rank_1": {"rank": 1, "alpha": 1.0},
        "rank_2": {"rank": 2, "alpha": 2.0},
        "rank_4": {"rank": min(4, base.input_dim, base.output_dim), "alpha": float(min(4, base.input_dim, base.output_dim))},
        "alpha_2r": {"rank": base.rank, "alpha": float(2 * base.rank)},
    }
    results = {}
    for name, changes in variants.items():
        payload = base.model_dump()
        payload.update(changes)
        config = LoraExperimentConfig.model_validate(payload)
        run_id = store.create("lora", f"ablation:{name}", config.model_dump())
        result = train(config, store, run_id)
        results[name] = {"run_id": run_id, "metrics": result["metrics"], "parameters": result["parameter_reports"]["lora"]}
    return results
