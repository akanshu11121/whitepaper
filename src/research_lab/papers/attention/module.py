"""Paper module facade: UI/API access is delegated to this independently testable class."""

import json
from typing import Any

import torch
import yaml

from research_lab.artifacts import ArtifactStore
from research_lab.config import BenchmarkConfig, ExperimentConfig
from research_lab.datasets import EOS, SequenceDataset, Vocabulary, build_dataset, pad
from research_lab.evaluation import sequence_metrics
from research_lab.papers.attention.benchmark import benchmark
from research_lab.papers.attention.inference import generate
from research_lab.papers.attention.runner import load_model, train
from research_lab.runtime import ROOT

CONTENT = ROOT / "papers" / "attention"


class AttentionModule:
    paper_id = "attention"

    def __init__(self, store: ArtifactStore | None = None):
        self.store = store or ArtifactStore()

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.paper_id, "title": "Attention Is All You Need",
            "year": 2017, "domain": "NLP", "task": "Sequence transduction",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
            "status": "implemented-local-validation", "readiness": 80,
            "assessment_note": "Engineering rubric, not code coverage or scientific certification; mean of documented dimensions.",
            "readiness_dimensions": {"understanding": 95, "mathematics": 92, "fidelity": 86, "experiments": 68, "benchmarks": 78, "visualization": 75, "testing": 82, "documentation": 84, "reproducibility": 70, "deployment": 70},
            "official_url": "https://arxiv.org/abs/1706.03762",
            "code_url": "https://github.com/tensorflow/tensor2tensor",
            "reported_results": {"wmt14_en_de_base_bleu": 27.3, "wmt14_en_de_big_bleu": 28.4, "wmt14_en_fr_base_bleu": 38.1, "wmt14_en_fr_big_bleu": 41.8, "wmt14_en_fr_section_6_1_bleu": 41.0},
        }

    def explain(self, level: int) -> list[dict[str, Any]]:
        if not 0 <= level <= 5:
            raise ValueError("level must be in [0, 5]")
        content = yaml.safe_load((CONTENT / "content.yaml").read_text())
        return [{"title": concept["title"], "body": concept["explanations"][level], "section": concept["section"], "source": concept["source"], "level": level} for concept in content["concepts"]]

    def content(self) -> dict[str, Any]:
        return yaml.safe_load((CONTENT / "content.yaml").read_text())

    def validate(self) -> dict[str, Any]:
        report = CONTENT / "VALIDATION.md"
        return {"status": "validated", "report": str(report.relative_to(CONTENT.parents[1])), "checks": [
            "paper PDF supplied and parsed", "equation/topology review completed", "ambiguities register recorded",
            "full WMT reproduction not claimed", "author result discrepancies preserved",
        ]}

    def preprocess(self, config: ExperimentConfig) -> SequenceDataset:
        return build_dataset(config)

    def train(self, config: ExperimentConfig, run_id: str) -> dict[str, Any]:
        dataset = self.preprocess(config)
        return train(config, dataset, self.store, run_id)

    def predict(self, run_id: str, text: str, beam_size: int = 1) -> dict[str, Any]:
        result = self.store.read(run_id)
        config = ExperimentConfig.model_validate(result["config"])
        vocab_tokens = json.loads((self.store.path(run_id) / "vocab.json").read_text())["tokens"]
        vocabulary = Vocabulary(vocab_tokens)
        if not text.strip() or len(text.split()) > config.max_length:
            raise ValueError("text must be nonempty and fit configured max_length")
        model = load_model(self.store, run_id, config, len(vocab_tokens), "cpu")
        source = pad([vocabulary.encode(text)])
        tokens, confidences = generate(model, source, min(config.model.max_length - 1, config.max_length + 4), beam_size)
        # Capture the prefix that actually produced the generated tokens, not a reference target.
        prefix = torch.tensor([tokens[0][:-1]], dtype=torch.long)
        with torch.inference_mode():
            logits, trace = model(source, prefix, capture=True)
        attention = {key: [tensor[0].tolist() for tensor in values] for key, values in trace.items() if key in {"encoder", "decoder", "cross"}}
        states = {key: [tensor[0].tolist() for tensor in values] for key, values in trace.items() if "states" in key}
        return {"input": text, "tokens": vocabulary.decode(tokens[0]), "confidence": confidences[0],
                "confidence_note": "Emitted conditional probability after blocking BOS/PAD; uncalibrated, includes EOS when emitted.",
                "attention": attention, "states": states,
                "source_labels": [vocab_tokens[token] for token in source[0].tolist()],
                "target_labels": [vocab_tokens[token] for token in prefix[0].tolist()],
                "unknown_tokens": [word for word in text.split() if word not in vocab_tokens],
                "terminated_with_eos": tokens[0][-1] == EOS,
                "source_ids": source[0].tolist(), "target_ids": prefix[0].tolist(),
                "logits": logits[0].tolist(), "model_type": config.model_type}

    def evaluate(self, predictions: list[list[str]], targets: list[list[str]]) -> dict[str, float]:
        return sequence_metrics(predictions, targets)

    def benchmark(self, config: BenchmarkConfig) -> dict[str, Any]:
        return benchmark(config)

    def visualize(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"type": "training-and-attention", "history": result.get("history", []), "attention_available": bool(result.get("attention"))}

    def run_experiment(self, config: ExperimentConfig, run_id: str) -> dict[str, Any]:
        return self.train(config, run_id)

    def get_examples(self) -> list[dict[str, Any]]:
        return [{"source": "0 1 2 3", "target": "3 2 1 0", "task": "reverse"}, {"source": "2 5 1", "target": "1 5 2", "task": "reverse"}]

    def get_limitations(self) -> list[str]:
        return ["Full WMT14 training and BLEU reproduction is not implemented.", "Full attention is quadratic in sequence length.", "Autoregressive generation remains sequential and this reference path does not use a KV cache.", "Attention weights are exposed as computation diagnostics, not guaranteed explanations.", "Local whitespace tokenization is not the paper's BPE/word-piece pipeline."]

    def get_references(self) -> list[dict[str, Any]]:
        return [{"title": "Attention Is All You Need", "authors": "Vaswani et al.", "venue": "NeurIPS 2017", "url": "https://arxiv.org/abs/1706.03762"}, {"title": "Tensor2Tensor", "authors": "Vaswani et al.", "url": "https://github.com/tensorflow/tensor2tensor", "license": "Apache-2.0"}]
