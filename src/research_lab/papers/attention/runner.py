"""Train/evaluate local experiments with persisted scientific provenance."""

import json
import math
import time
from typing import Any

import torch
from sacrebleu.metrics import BLEU
from safetensors.torch import load_model as load_safe_model
from safetensors.torch import save_model

from research_lab.artifacts import ArtifactStore
from research_lab.config import ExperimentConfig
from research_lab.datasets import PAD, SequenceDataset
from research_lab.evaluation import sequence_metrics
from research_lab.papers.attention.baseline import GRUSeq2Seq
from research_lab.papers.attention.inference import GenerationModel, generate
from research_lab.papers.attention.model import Transformer
from research_lab.papers.attention.training import nll_loss, noam_rate, smoothed_loss
from research_lab.runtime import environment, seed_all


def build_model(vocab_size: int, config: ExperimentConfig) -> Transformer | GRUSeq2Seq:
    if config.model_type == "gru":
        return GRUSeq2Seq(vocab_size, config.model)
    return Transformer(vocab_size, config.model)


def _batches(dataset: SequenceDataset, split: str, batch_size: int, seed: int):
    values = list(getattr(dataset, split))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist() if split == "train" else list(range(len(values)))
    for start in range(0, len(values), batch_size):
        pairs = [values[i] for i in order[start:start + batch_size]]
        if pairs:
            yield dataset.batch(pairs)


@torch.no_grad()
def evaluate(model: GenerationModel, dataset: SequenceDataset, split: str, device: str,
             batch_size: int, max_tokens: int) -> dict[str, Any]:
    model.eval()
    total_loss, count = 0.0, 0
    predictions, targets = [], []
    for source, target in _batches(dataset, split, batch_size, 12345):
        source, target = source.to(device), target.to(device)
        logits, _ = model(source, target[:, :-1])
        loss = nll_loss(logits, target[:, 1:])
        total_loss += float(loss) * int(target[:, 1:].ne(PAD).sum())
        count += int(target[:, 1:].ne(PAD).sum())
        generated, _ = generate(model, source, max_tokens)
        predictions.extend([dataset.vocab.decode(row) for row in generated])
        targets.extend([dataset.vocab.decode(row.tolist()) for row in target])
    metrics = sequence_metrics(predictions, targets)
    metrics.update({"nll": total_loss / max(count, 1), "perplexity": math.exp(min(700, total_loss / max(count, 1)))})
    return {"metrics": metrics, "predictions": predictions, "targets": targets}


def train(config: ExperimentConfig, dataset: SequenceDataset, store: ArtifactStore,
          run_id: str, device: str | None = None) -> dict[str, Any]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    device = device or config.device
    seed_all(config.seed)
    store.write(run_id, "dataset.json", {**dataset.statistics(), "splits": dataset.serialize()})
    store.write(run_id, "vocab.json", {"tokens": dataset.vocab.tokens})
    store.update(run_id, status="running", environment=environment(), dataset=dataset.statistics())
    model = build_model(len(dataset.vocab.tokens), config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        source, target = next(_batches(dataset, "train", config.batch_size, config.seed + step))
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(source, target[:, :-1])
        loss = smoothed_loss(logits, target[:, 1:], config.smoothing) if config.smoothing else nll_loss(logits, target[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rate = noam_rate(step, config.model.d_model, config.warmup_steps, config.learning_rate_factor)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if step == 1 or step % max(1, config.steps // 10) == 0 or step == config.steps:
            history.append({"step": float(step), "loss": float(loss.detach()), "learning_rate": rate})
            store.update(run_id, history=history, progress={"step": step, "total": config.steps})
    elapsed = time.perf_counter() - start
    max_tokens = min(config.model.max_length - 1, config.max_length + 4)
    validation = evaluate(model, dataset, "validation", device, config.batch_size, max_tokens)
    test = evaluate(model, dataset, "test", device, config.batch_size, max_tokens)
    model_path = store.path(run_id) / "model.safetensors"
    save_model(model, str(model_path))
    copy_metrics = sequence_metrics([pair.source.split() for pair in dataset.test], test["targets"])
    majority = dataset.majority_baseline()
    majority_metrics = sequence_metrics([[majority] * len(pair.source.split()) for pair in dataset.test], test["targets"])
    examples = [{"source": pair.source, "target": pair.target, "prediction": prediction,
                 "correct": prediction == pair.target.split()}
                for pair, prediction in zip(dataset.test, test["predictions"], strict=True)]
    failures = [{**row, "likely_reason": "generation error after bounded training; insufficient evidence to attribute causally",
                 "component": "autoregressive decoder / learned representations", "mitigation": "more data/updates; compare validation curves, positions and beam search"}
                for row in examples if not row["correct"]]
    bleu = None
    if config.task == "parallel":
        scorer = BLEU()
        score = scorer.corpus_score([" ".join(row) for row in test["predictions"]], [[" ".join(row) for row in test["targets"]]])
        bleu = {"score": score.score, "signature": str(scorer.get_signature()), "comparable_to_paper": False}
    result = {
        "status": "completed", "device": device, "duration_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history, "validation": validation["metrics"], "test": test["metrics"],
        "examples": examples, "failures": failures,
        "baselines": {"copy_input": copy_metrics, "majority_token": majority_metrics}, "bleu": bleu,
        "environment": environment(), "dataset": dataset.statistics(),
        "model_artifact": "model.safetensors", "scientific_status": "local-derived-result",
    }
    store.update(run_id, **result)
    return result


def load_vocab(store: ArtifactStore, run_id: str) -> list[str]:
    return json.loads((store.path(run_id) / "vocab.json").read_text())["tokens"]


def load_model(store: ArtifactStore, run_id: str, config: ExperimentConfig, vocab_size: int, device: str) -> GenerationModel:
    model = build_model(vocab_size, config)
    model.to(device)
    load_safe_model(model, str(store.path(run_id) / "model.safetensors"), strict=True)
    return model
