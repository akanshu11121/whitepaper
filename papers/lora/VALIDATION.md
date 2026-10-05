# LoRA Paper Validation Report

## Identification and evidence

**LoRA: Low-Rank Adaptation of Large Language Models**, Edward J. Hu, Yelong Shen,
Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
Microsoft Corporation. arXiv **2106.09685v2**, 16 October 2021; published at ICLR
2022. DOI was not verified and is not invented.

The requested local file `papers/LoRA.pdf` was not present at implementation start.
The paper was validated from the official arXiv HTML/PDF URL supplied by the user.

Verified sources:

- https://arxiv.org/abs/2106.09685
- https://arxiv.org/html/2106.09685
- https://openreview.net/forum?id=nZeVKeeFYf9
- https://github.com/microsoft/LoRA

## What the paper proposes

LoRA is a **parameter-efficient adaptation algorithm / reparameterization**, not a
new base language model, dataset, tokenizer, evaluation metric, or optimizer.

For a frozen pretrained weight matrix `W₀ ∈ R^(d×k)`, LoRA constrains the update:

```text
W = W₀ + (α / r) B A
A ∈ R^(r×k), B ∈ R^(d×r), r << min(d, k)
```

The forward pass is:

```text
h = W₀x + (α / r) B A x
```

The paper initializes `A` with a random Gaussian distribution and `B` to zero,
so the initial adaptation is exactly zero. The base parameters are frozen and
only the low-rank factors are optimized. At deployment, `BA` can be merged into
`W₀` so the merged model has no extra inference layer.

The authors focus primarily on Transformer attention weights, especially query
and value projections, while leaving MLP, LayerNorm and bias adaptation for
future investigation in their main setup.

## Reproducibility classification

| Component | Classification | Evidence / boundary |
|---|---|---|
| Low-rank update equation | Reproducible | §4.1 Eq. 3 and dimensions |
| Frozen base / trainable A and B | Reproducible | §4.1 and official loralib |
| Gaussian A / zero B initialization | Reproducible | §4.1 |
| α/r scaling | Reproducible | §4.1 |
| Merge/unmerge at inference | Reproducible | §4.1 and official implementation |
| Attention target-module selection | Partially reproducible | §4.2 and §7.1 describe studied choices, but architecture-specific wiring varies |
| GLUE/RoBERTa/DeBERTa results | Partially reproducible | checkpoints, data, seeds and historical scripts required |
| GPT-2 E2E/DART/WebNLG results | Partially reproducible | official examples exist, but full environment and data versioning are external |
| GPT-3 175B results | Not reproducible locally | inaccessible model-scale hardware/checkpoint and task pipeline |
| Exact paper confidence intervals | Not reproducible from paper alone | seed schedules and all run artifacts unavailable |

Paper-reported headline claims include up to 10,000× fewer trainable parameters
and up to 3× lower GPU memory in the GPT-3 comparison. These remain **author-
reported**, not measurements from this repository.

## ASSUMPTIONS & AMBIGUITIES REGISTER

| ID | Paper says / unclear detail | Engineering interpretation | Alternative | Impact |
|---|---|---|---|---|
| L1 | A random Gaussian initializer is specified but its exact standard deviation is not | Use `std=0.02`, configurable, matching common Transformer-scale initialization | PyTorch default / variance-scaled init | Affects early adapter gradients and local trajectories |
| L2 | `α` is described as a constant often set to the first tested rank | Expose alpha; default `alpha=rank` and report effective scale | Tune alpha independently | Changes update magnitude while holding rank fixed |
| L3 | “Apply to attention weights” varies by architecture | Local synthetic model adapts one dense projection; metadata supports q/k/v/o labels | Adapt q and v in a Transformer | Synthetic task validates equation, not module-selection quality |
| L4 | Official implementation merges on `eval()`; merge policy is framework-specific | Make `merge()` and `unmerge()` explicit and test idempotence | Auto-merge on eval/train transitions | Explicit calls avoid hidden parameter mutation in research experiments |
| L5 | Paper baselines use external datasets/checkpoints | Implement full fine-tuning and frozen-base baselines on a controlled regression task | Download GLUE/GPT-2 | Quality values are not comparable to paper tables |
| L6 | Bias training is discussed as an optional extension | Default bias frozen; `bias_mode` is recorded and configurable for local experiment | Train all biases | Keeps central LoRA claim isolated |
| L7 | Rank can equal or exceed matrix dimensions in theory, but LoRA assumes low rank | Reject `r > min(out_features,in_features)` | Allow redundant factors | Prevents misleading “low-rank” configuration |
| L8 | Low-rank regression target is not a language-model task | Generate target weight as a known low-rank shift and evaluate adaptation error | Use a pretrained text model | Gives a controlled test of the proposed parameterization without fabricated NLP results |

### Missing-information format

MISSING FROM PAPER: exact Gaussian standard deviation, all random seeds, complete
checkpoint/dataset hashes, and enough GPT-3 infrastructure to rerun the largest study.

ENGINEERING ASSUMPTION: L1–L8 define bounded local experiments and do not alter the
paper claims.

WHY THIS ASSUMPTION: it makes gradient isolation, rank, scaling, merge equivalence,
and parameter savings directly measurable on CPU.

IMPACT: local results validate implementation properties only; they do not reproduce
GLUE, GPT-2, or GPT-3 numbers.

ALTERNATIVE: install official `loralib`, obtain licensed checkpoints/data, pin the
historical environment, and run the official examples with a complete artifact log.

## Research decomposition

Problem → full fine-tuning duplicates huge checkpoints and optimizer state;
motivation → many downstream tasks need compact task-specific deltas;
hypothesis → adaptation updates occupy a low intrinsic-rank subspace;
method → freeze `W₀`, optimize `A/B`, scale by `α/r`, merge for inference;
architecture → LoRA modules in selected Transformer dense projections;
training → task loss and Adam on adapter parameters only;
inference → unmerged parallel path or merged `W₀ + BA` path;
evaluation → quality, trainable count, memory, throughput and latency;
ablations → rank, target matrices, scaling, bias and task size;
limitations → target-module choice, rank budget, task batching with separate adapters,
and the empirical rather than universal nature of low-rank updates.

## Implementation strategy

- PyTorch research logic is isolated under `src/research_lab/papers/lora/`.
- `LoRALinear` is a paper-faithful reference layer with explicit merge/unmerge.
- Adapter state is persisted separately using safetensors; base weights remain reusable.
- A controlled low-rank linear-regression adaptation task validates the mechanism.
- Full fine-tuning and frozen-base baselines are measured on exactly the same data.
- Rank and alpha ablations reuse identical generated data and seeds.
- Metrics include MSE, parameter counts, trainable fraction, adapter bytes, merge
  equivalence, and measured unmerged/merged latency.
- The Netlify browser mode uses a clearly labeled educational low-rank matrix demo.

## Scientific status

| Result class | Meaning in this module |
|---|---|
| Paper result | Values reported in LoRA Tables 2–4 and headline claims |
| Reproduced result | Not claimed for the original NLP benchmarks |
| Derived result | Local controlled regression and benchmark values, generated at runtime |
| Engineering estimate | Parameter/byte formulas before runtime measurement |
