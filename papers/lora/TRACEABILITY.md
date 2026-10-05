# LoRA Paper ↔ Code Traceability

| Paper component | Section / equation | Implementation | Test | Experiment |
|---|---|---|---|---|
| Frozen base update | §4.1 Eq. 3 | `layers.py:LoRALinear.forward` | `test_lora.py::test_forward_equation` | all local runs |
| Low-rank factors | §4.1 | `layers.py:LoRALinear.lora_A/lora_B` | shape/rank tests | rank ablation |
| A random / B zero init | §4.1 | `layers.py:reset_lora_parameters` | zero-update test | initial-loss comparison |
| α/r scaling | §4.1 | `layers.py:scaling` | scaling test | alpha ablation |
| Frozen W₀ | §4.1 | `layers.py:mark_only_lora_trainable` | gradient isolation | LoRA vs full FT |
| Transformer target choice | §4.2/§7.1 | `config.py:target` metadata | config validation | target-module extension point |
| Merge for no latency | §4.1 | `layers.py:merge/unmerge` | output equivalence | merged/unmerged benchmark |
| Adapter-only checkpoint | official loralib pattern | `layers.py:lora_state_dict` | save/load test | artifact persistence |
| Trainable parameter formula | §5.1 | `layers.py:parameter_report` | count oracle | rank sweep |
| Full fine-tuning baseline | §5.1 | `runner.py:train_full` | baseline integration | same regression task |
| Frozen-base baseline | engineering control | `runner.py:train_frozen` | baseline integration | same regression task |
| GLUE/GPT-2/GPT-3 results | §5 | documentation only | N/A | NOT IMPLEMENTED locally |
