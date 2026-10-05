# Attention Is All You Need — module

## Summary

The Transformer replaces sequence-aligned recurrence and convolution with stacks
of multi-head self-attention and position-wise feed-forward networks. The encoder
builds contextual source representations; the causal decoder generates a target
sequence one token at a time and uses encoder–decoder attention.

## Reference configuration

The paper's base model is N=6, `d_model=512`, `d_ff=2048`, h=8, `d_k=d_v=64`,
dropout=.1, label smoothing=.1, shared 37K BPE vocabulary, 100K steps, 8 P100 GPUs,
25K source and target tokens per batch, Adam (β₁=.9, β₂=.98, ε=1e−9), and 4K warmup.
The local default is deliberately smaller and bounded for laptop experiments.

## Local experiment

The UI or API trains a Transformer on disjoint synthetic copy/reversal sequences.
Use free-running exact match and token accuracy to evaluate; NLL/perplexity is
teacher-forced. The GRU is a labeled engineering baseline. `ablations.py` provides
single-head, no-smoothing, positional and FFN variants.

## Visual exploration

The interface exposes progressive explanations, equations, architecture flow,
training history, inference, limitations and traceability. Attention tensors are
available when capture mode is requested by research code; their presence is not
treated as proof of causal explanation.

## Dataset and licensing

The paper used WMT14 English–German (~4.5M sentence pairs) and English–French
(36M), plus WSJ/related corpora for parsing. This project does not redistribute
those datasets. The executable local adapter accepts user-authorized text pairs,
uses whitespace tokenization and a train-only joint vocabulary, and records a hash.

## Results policy

Reported paper numbers are documented in `VALIDATION.md` and API metadata. Local
results are persisted as `local-derived-result` with hardware/software evidence.
No exact paper reproduction is claimed.

## References and credits

- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS 30,
  https://arxiv.org/abs/1706.03762
- Official/author code: Tensor2Tensor, Apache-2.0,
  https://github.com/tensorflow/tensor2tensor
- PyTorch, FastAPI, React, Vite, safetensors, NumPy, PyYAML, psutil, SacreBLEU:
  each remains governed by its upstream license.
