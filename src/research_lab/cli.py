"""Command line entry points for local validation and experiments."""

import argparse
import json

from research_lab.api import app
from research_lab.config import BenchmarkConfig, ExperimentConfig
from research_lab.papers.attention.benchmark import benchmark
from research_lab.papers.attention.module import AttentionModule


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Research Implementation Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("validate")
    sub.add_parser("benchmark")
    sub.add_parser("experiment")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif args.command == "validate":
        print(json.dumps(AttentionModule().validate(), indent=2))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(BenchmarkConfig(seed=42, length=32, batch_size=4, d_model=64, heads=4, repeats=30, warmup=5)), indent=2))
    else:
        module = AttentionModule()
        config = ExperimentConfig(seed=42, steps=100, train_size=128, eval_size=16, batch_size=32, min_length=3, max_length=8, symbols=12, warmup_steps=100, learning_rate_factor=0.5, smoothing=0.1, pairs=None)
        run_id = module.store.create("attention", "experiment", config.model_dump())
        print(json.dumps(module.train(config, run_id), indent=2))
