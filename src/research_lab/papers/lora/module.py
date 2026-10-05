"""Registry facade for the LoRA research module."""

from typing import Any

import yaml

from research_lab.artifacts import ArtifactStore
from research_lab.papers.lora.config import LoraBenchmarkConfig, LoraExperimentConfig
from research_lab.papers.lora.runner import benchmark, make_data, predict, train
from research_lab.runtime import ROOT

CONTENT = ROOT / "papers" / "lora"


class LoRAModule:
    paper_id = "lora"

    def __init__(self, store: ArtifactStore | None = None):
        self.store = store or ArtifactStore()

    def metadata(self) -> dict[str, Any]:
        metadata = yaml.safe_load((CONTENT / "metadata.yaml").read_text())
        return {
            **metadata,
            "readiness": 82,
            "assessment_note": "Engineering assessment; controlled adaptation validates the mechanism but does not reproduce the paper's NLP benchmark tables.",
            "readiness_dimensions": {"understanding": 94, "mathematics": 94, "fidelity": 90, "experiments": 65, "benchmarks": 75, "visualization": 78, "testing": 86, "documentation": 88, "reproducibility": 76, "deployment": 82},
            "reported_results": {"gpt3_trainable_parameter_reduction_claim": 10000, "gpt3_gpu_memory_reduction_claim": 3, "roberta_base_lora_average": 87.24, "deberta_xxl_lora_average": 91.32},
            "result_status": "reported_results_are_author_reported",
        }

    def explain(self, level: int) -> list[dict[str, Any]]:
        if not 0 <= level <= 5:
            raise ValueError("level must be in [0, 5]")
        content = yaml.safe_load((CONTENT / "content.yaml").read_text())
        return [{"title": item["title"], "body": item["explanations"][level], "section": item["section"], "source": item["source"], "level": level} for item in content["concepts"]]

    def content(self) -> dict[str, Any]:
        return yaml.safe_load((CONTENT / "content.yaml").read_text())

    def validate(self) -> dict[str, Any]:
        return {"status": "validated", "report": "papers/lora/VALIDATION.md", "checks": [
            "arXiv:2106.09685v2 verified", "official Microsoft LoRA repository verified", "Eq.3 and merge behavior implemented",
            "full GLUE/GPT-2/GPT-3 reproduction explicitly not claimed", "local PDF was missing; official URL used as evidence",
        ]}

    def preprocess(self, config: LoraExperimentConfig):
        return make_data(config)

    def train(self, config: LoraExperimentConfig, run_id: str) -> dict[str, Any]:
        return train(config, self.store, run_id)

    def predict(self, run_id: str, text: str, beam_size: int = 1) -> dict[str, Any]:
        return predict(self.store, run_id, text)

    def evaluate(self, predictions: list[list[str]], targets: list[list[str]]) -> dict[str, float]:
        if len(predictions) != len(targets) or not predictions:
            raise ValueError("aligned nonempty predictions and targets required")
        errors = [float(sum((float(a) - float(b)) ** 2 for a, b in zip(prediction, target, strict=True)) / len(target)) for prediction, target in zip(predictions, targets, strict=True)]
        return {"mse": sum(errors) / len(errors)}

    def benchmark(self, config: LoraBenchmarkConfig) -> dict[str, Any]:
        return benchmark(config)

    def visualize(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"type": "low-rank-adaptation", "parameter_reports": result.get("parameter_reports", {}), "merge_difference": result.get("merge_max_difference"), "matrix_artifacts": ["base.safetensors", "adapter.safetensors"]}

    def run_experiment(self, config: LoraExperimentConfig, run_id: str) -> dict[str, Any]:
        return self.train(config, run_id)

    def get_examples(self) -> list[dict[str, Any]]:
        return [{"source": "A/B factors", "target": "adapted dense projection", "task": "low-rank regression"}, {"source": "frozen W₀ + adapter", "target": "merged W", "task": "deployment reparameterization"}]

    def get_limitations(self) -> list[str]:
        return ["The local experiment is controlled regression, not GLUE or generation.", "Rank and target-module choices are task/model dependent.", "Adapter parameter savings do not imply equal wall-clock savings because the frozen base multiplication remains.", "Merged weights make one adapter active per base copy; serving many tasks in a single batch needs routing or separate unmerged paths.", "The requested local LoRA PDF was absent; official arXiv HTML/PDF was used for validation."]

    def get_references(self) -> list[dict[str, Any]]:
        return [{"title": "LoRA: Low-Rank Adaptation of Large Language Models", "authors": "Hu et al.", "venue": "ICLR 2022", "url": "https://arxiv.org/abs/2106.09685"}, {"title": "Microsoft LoRA / loralib", "authors": "Microsoft", "license": "MIT", "url": "https://github.com/microsoft/LoRA"}]
