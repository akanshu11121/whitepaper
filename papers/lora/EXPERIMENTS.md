# LoRA experiment protocol

## Controlled adaptation task

Generate a fixed base matrix `W₀` and a target matrix:

```text
W_target = W₀ + (α_target / r_target) B_target A_target
```

Sample input vectors and targets `y = W_target x`. Start all methods at `W₀`:

- Frozen base: no trainable update
- Full fine-tuning: train all entries of `W₀`
- LoRA: freeze `W₀`; train only `A/B`

This isolates whether a low-rank update can recover a known low-rank adaptation.
It is a mechanism test, not an NLP benchmark.

## Metrics

- held-out mean squared error
- trainable parameter count and fraction
- adapter-only bytes versus full model bytes
- merge/unmerge maximum absolute difference
- unmerged versus merged CPU latency
- failure and invalid-configuration records

## Ablations

- rank `r`: 1, 2, 4, 8 where dimensions permit
- alpha: `r`, `2r`, and fixed alpha
- LoRA dropout: zero and configured dropout
- target matrix: one projection versus multiple projections in extension experiments
- full fine-tuning versus frozen base

## Reproducibility

Every run stores config, seed, environment, generated-data fingerprint, model version,
metrics, adapter artifact and failure diagnostics. Values produced by this protocol
are labeled `local-derived-result`.

## Observed local-derived run

Environment: Python 3.13.9, PyTorch 2.14.1, macOS arm64, CPU, deterministic algorithms,
seed 42. Configuration: input 16, output 8, LoRA rank 2, target rank 2, 100 updates,
128 training vectors, 32 held-out vectors, batch size 32, learning rate .08.

| Method | Held-out MSE | Trainable parameters |
|---|---:|---:|
| Frozen base | 1.305006 | 0 |
| LoRA rank 2 | 0.000377 | 48 |
| Full fine-tuning | 0.000228 | 184 |

LoRA merged/unmerged held-out MSE was 0.000377 in both paths, with observed maximum
output difference `7.15e-7` in float32. This is a controlled low-rank regression result,
not a language-model benchmark or reproduction of a paper table.
