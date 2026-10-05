# AI Research Implementation Lab

An extensible, traceable paper-to-product workspace. The modules currently implement
**Attention Is All You Need** (Vaswani et al., NeurIPS 2017) and **LoRA: Low-Rank
Adaptation of Large Language Models** (Hu et al., ICLR 2022) as readable PyTorch
reference implementation with scientific tests, local experiments, benchmarks,
an API, and an interactive research UI.

## Quick start

Requires Python 3.11–3.13, Node 20+, and optionally Docker.

```bash
uv sync --dev
uv run pytest
uv run ruff check src tests
uv run research-lab serve
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API is `http://localhost:8000`.

Or run the combined production-shaped container:

```bash
docker compose up --build
open http://localhost:8000
```

## What is implemented

- §3 scaled dot-product attention, multi-head attention, encoder/decoder stacks,
  causal masks, post-normalized residuals, FFNs, tied/scaled embeddings and
  sinusoidal/learned/no positional variants.
- §5 Adam settings, Noam learning-rate schedule, residual dropout and explicit
  padding-aware label smoothing.
- Safe synthetic copy/reversal tasks and a user-supplied parallel-text adapter.
- Teacher-forced NLL/perplexity plus free-running exact match/token accuracy.
- Greedy and readable beam search, safetensors checkpoints, reproducibility metadata,
  attention capture, resource benchmark, GRU engineering baseline, and ablation presets.
- Registry-driven FastAPI routes and responsive React research interface.
- LoRA frozen-base low-rank adapters, merge/unmerge, adapter-only safetensors,
  full/frozen baselines, controlled regression, rank/alpha ablations and merged
  versus unmerged benchmarks.

## Scientific boundary

The paper's WMT14 scores are displayed as **author-reported**. This repository does
not claim exact WMT reproduction: the historical BPE/word-piece recipe, complete
filtering/scoring details, seed, initializer and original hardware environment are
not specified sufficiently by the paper. See [`papers/attention/VALIDATION.md`](papers/attention/VALIDATION.md)
for the validation report, assumptions register, evidence classification and the
reported 41.8 versus 41.0 English–French discrepancy.

## API

```text
GET  /api/papers
GET  /api/papers/{paper_id}
GET  /api/papers/{paper_id}/explanation?level=5
GET  /api/papers/{paper_id}/validation
GET  /api/papers/{paper_id}/references
GET  /api/papers/{paper_id}/limitations
POST /api/papers/{paper_id}/experiment
POST /api/papers/{paper_id}/predict
POST /api/papers/{paper_id}/benchmark
GET  /api/papers/{paper_id}/results
GET  /api/papers/{paper_id}/results/{run_id}
GET  /api/metrics
```

Experiment requests are bounded by Pydantic configuration and execute through a
single worker. Run artifacts are stored under `runs/`; model weights use safetensors,
not pickle. Set `LAB_RUNS_DIR` to use a persistent volume.

Security and operations guidance is in [`SECURITY.md`](SECURITY.md) and
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Configure `LAB_API_TOKEN` before exposing
the service beyond a trusted local environment.

## Architecture

```text
registry.yaml → module facade → research logic → safe artifacts
       ↓               ↓              ↓
    FastAPI         React UI       tests/benchmarks
```

Research logic is separated from transport, presentation and persistence. New papers
register an allowlisted module implementing the contract in `research_lab/core.py`.
The extension guide is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Reproducibility

Each run persists the config, seed, source and dataset hashes, environment, metrics,
examples and checkpoint. PyTorch deterministic algorithms are enabled where feasible;
bitwise reproducibility is deliberately **not** claimed across hardware/software stacks.

## License and credits

Project code is MIT-licensed (see [`LICENSE`](LICENSE)). The paper is attributed to
its authors and NeurIPS; the supplied PDF is retained as user-provided research
material. Tensor2Tensor is an archived Apache-2.0 repository. Datasets supplied by
users retain their own licenses and are not redistributed here.
