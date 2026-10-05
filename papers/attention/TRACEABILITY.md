# Paper ↔ Code Traceability

All paths are relative to the repository; implementation files live under
`src/research_lab/papers/attention/` unless specified.

| Paper component | Section / equation | Implementation | Validation | Experiment |
|---|---|---|---|---|
| Scaled attention | §3.2.1 Eq.1 | `attention.py:scaled_attention` | `tests/test_science.py` numerical oracle, masks, gradient | `benchmark` reference vs SDPA |
| Multi-head attention | §3.2.2 | `attention.py:MultiHeadAttention` | shape and SDPA equivalence | single-head ablation |
| Encoder post-norm | §3.1 | `model.py:EncoderLayer` | padding invariance | copy/reversal |
| Causal decoder/cross-attention | §3.1/3.2.3 | `model.py:DecoderLayer` | future-token invariance | free-running generation |
| Positionwise FFN | §3.3 Eq.2 | `model.py:feed_forward` | independent position shape | trained model |
| Tied/scaled embeddings | §3.4 | `model.py:Transformer` | parameter identity | tied-output training |
| Sinusoidal positions | §3.5 | `model.py:sinusoidal_positions` | explicit values | no-position/learned ablations |
| Residual + LayerNorm | §3.1/5.4 | `model.py:EncoderLayer/DecoderLayer` | finite outputs and gradients | dropout control |
| Adam / schedule | §5.3 Eq.3 | `training.py:noam_rate` | warmup and decay | training curve |
| Label smoothing | §5.4 | `training.py:smoothed_loss` | probability normalization/pad exclusion | no-smoothing ablation |
| Autoregression/beam | §6.1 | `inference.py:generate` | EOS/beam integration | prediction UI |
| Checkpoint averaging | §6.1 | `inference.py:average_checkpoints` | incompatible state rejection | CLI average |
| WMT experiments | §5.1/6.1 | local text adapter, base/big presets | adapter tests only | **NOT IMPLEMENTED: full WMT reproduction** |
| Table 3 variations | §6.2 | `runner.py:run_ablations` | separate seeded retraining | CLI ablate |
| Constituency parsing | §6.3 | no parser-specific pipeline | none | **NOT IMPLEMENTED** |

Engineering components: dataset adapter → `datasets.py`; persistence → `artifacts.py`;
module contract → `core.py`; registry → `registry.py`; API → `api.py`;
browser → `frontend/src/`; CLI → `cli.py`; scientific tests → `tests/`.
