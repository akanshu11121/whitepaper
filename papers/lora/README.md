# LoRA: Low-Rank Adaptation of Large Language Models

## Core idea

Freeze the pretrained matrix `W₀` and learn a task-specific low-rank update:

```text
W = W₀ + (α / r) B A
```

Only `A` and `B` are trainable. At deployment, the update can be merged into
`W₀`, recovering an ordinary dense linear layer with no extra adapter depth.

## What this module implements

- Paper-faithful PyTorch `LoRALinear`
- Explicit frozen-base and trainable-adapter behavior
- Configurable rank, alpha, dropout and bias policy
- Explicit merge/unmerge and adapter-only safetensors artifacts
- Full fine-tuning and frozen-base local baselines
- Controlled low-rank regression adaptation experiment
- Rank, alpha and merge/latency benchmark paths
- Interactive browser visualization for Netlify deployment

## What this module does not claim

The local experiment does not reproduce the paper's GLUE, GPT-2, GPT-3, WikiSQL,
MNLI or SAMSum results. The paper's datasets, pretrained checkpoints, original
training scripts and GPT-3 hardware are not redistributed here. See `VALIDATION.md`.

## References and credits

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022,
  https://arxiv.org/abs/2106.09685
- Official implementation: Microsoft `loralib`, MIT,
  https://github.com/microsoft/LoRA
- This module uses PyTorch and safetensors under their upstream licenses.
