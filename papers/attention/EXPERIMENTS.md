# Experiments, baselines, and benchmark protocol

## Paper-faithful reference configuration

The documented paper base is N=6, `d_model=512`, `d_ff=2048`, h=8,
`d_k=d_v=64`, dropout=.1, label smoothing=.1, 37K shared BPE vocabulary,
100K updates, ~25K source and target tokens per batch, Adam β₁=.9, β₂=.98,
ε=1e−9 and 4K warmup. The local API bounds width/depth for laptop safety;
it does not silently pretend to run this historical configuration.

## Executable local protocol

1. Choose `copy`, `reverse`, or authorized `parallel` pairs.
2. Generate or validate disjoint train/validation/test sources. The vocabulary is
   fitted on training pairs only; a dataset SHA-256 is stored.
3. Seed Python, NumPy, PyTorch and CUDA where available; enable deterministic
   algorithms where supported.
4. Train with shifted teacher forcing, pad-aware loss, Adam and Eq. 3 schedule.
5. Save config, source/dataset hashes, environment, history, safetensors, metrics,
   examples, failures and baseline values under the run ID.
6. Evaluate teacher-forced unsmoothed NLL/perplexity and free-running exact match/
   token accuracy. Uploaded parallel data also receives a SacreBLEU score and signature.

Run one experiment from the CLI:

```bash
uv run research-lab experiment
```

Or use `POST /api/papers/attention/experiment`; the API returns a run ID and the
single worker persists progress for polling.

## Baselines and ablations

- `copy_input`: a diagnostic upper-bound on copy tasks, not a learned model.
- `majority_token`: emits the most frequent training target token, measuring task
  imbalance rather than sequence competence.
- `GRUSeq2Seq`: a small engineering recurrent baseline for local comparison; it is
  not the paper's historical GNMT or a claim of an equivalent RNN baseline.
- `ablations.py`: full model, single head, no smoothing, learned/no positions, and
  smaller FFN. Each variant receives a separate run ID and identical base seed/data
  generation where applicable. Table 3's independent `d_k/d_v` sweeps are not
  silently represented by the equal-width local presets.

## Systems benchmark

`POST /api/papers/attention/benchmark` warms up and repeats reference Python
attention and PyTorch SDPA on the CPU, reporting p50/p95/mean latency, throughput,
process RSS snapshot, parameter bytes, analytical attention-score tensor size and
maximum numerical difference. GPU/VRAM, money and peak isolated allocation are
reported unavailable when not measured. Results are environment-specific.

## Observed smoke evidence

These are local-derived smoke measurements, not paper results:

| Run | Environment | Configuration | Result |
|---|---|---|---|
| Transformer smoke | macOS arm64, Python 3.13.9, PyTorch 2.14.1, CPU, 2 threads | reverse, d=16, h=4, 1 layer, 12 updates, 32 train / 8 test | exact match 0.000, token accuracy 0.136, NLL 1.804, perplexity 6.075 |
| Transformer attention capture | same environment | copy, d=16, h=4, 1 layer, 8 updates | encoder/decoder/cross maps captured; exact match 0.000, token accuracy 0.094 |
| GRU engineering baseline | same environment | copy, d=16, 1 layer, 4 updates | exact match 0.000, token accuracy 0.167, NLL 1.680, perplexity 5.363 |
| Attention benchmark | same environment | B=2, L=8, d=16, h=4, 5 repeats | reference p50 0.0507ms; SDPA p50 0.0280ms; max difference measured by endpoint |

Short runs intentionally do not establish quality. The observed values are retained
to prove the pipeline executes and are not compared to WMT BLEU.

## Failure analysis

Every failed held-out example is stored with source, target, prediction, component
label and a cautious mitigation note. The default note avoids claiming whether a
failure is caused by the paper or implementation without controlled evidence.
