"""Repeated forward-only latency and numerical equivalence for reference vs SDPA."""

import time
from typing import Any

import numpy as np
import psutil
import torch

from research_lab.config import BenchmarkConfig
from research_lab.papers.attention.attention import MultiHeadAttention
from research_lab.runtime import environment, seed_all


@torch.inference_mode()
def benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    seed_all(config.seed)
    device = "cpu"  # CPU path is stable and comparable across browser benchmark requests.
    start = time.perf_counter()
    q = torch.randn(config.batch_size, config.length, config.d_model, device=device)
    reference = MultiHeadAttention(config.d_model, config.heads).to(device).eval()
    optimized = MultiHeadAttention(config.d_model, config.heads, optimized=True).to(device).eval()
    optimized.load_state_dict(reference.state_dict())
    startup_ms = (time.perf_counter() - start) * 1000
    paths = {}
    for name, module in (("reference", reference), ("sdpa", optimized)):
        for _ in range(config.warmup):
            module(q, q, q)
        samples = []
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        for _ in range(config.repeats):
            start = time.perf_counter()
            module(q, q, q)
            samples.append((time.perf_counter() - start) * 1000)
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        paths[name] = {
            "latency_ms": {"p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95)), "mean": float(np.mean(samples))},
            "throughput_sequences_per_second": config.batch_size / (float(np.mean(samples)) / 1000),
            "cpu_core_equivalents": cpu_seconds / max(wall_seconds, 1e-9), "samples_ms": samples,
        }
    expected, _ = reference(q, q, q)
    actual, _ = optimized(q, q, q)
    params = sum(parameter.numel() for parameter in reference.parameters())
    return {
        "status": "completed", "device": device, "config": config.model_dump(),
        "parameter_count": params, "parameter_bytes_fp32": params * 4,
        "startup_ms": startup_ms, "process_rss_bytes": psutil.Process().memory_info().rss,
        "rss_scope": "whole process snapshot, not incremental model allocation or peak",
        "attention_matrix_bytes_fp32": config.batch_size * config.heads * config.length**2 * 4,
        "matrix_memory_kind": "analytical tensor-size calculation; not measured peak",
        "paths": paths, "max_absolute_difference": float((expected - actual).abs().max()),
        "environment": environment(), "scientific_status": "engineering-measurement",
        "unavailable": ["GPU/VRAM", "monetary cost", "isolated peak RAM"],
    }
