# Paper Validation Report

## Identification and evidence

**Attention Is All You Need**, Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.
Google Brain / Google Research / University of Toronto (work affiliations as stated).
NeurIPS 2017, Advances in Neural Information Processing Systems 30.
Input: `papers/Attention_is_all_you_need.pdf`, 15 pages, arXiv **1706.03762v7**,
2 August 2023; original submission 2017. The implementation targets this supplied version.
ArXiv identifier is used; no proceedings DOI was verified. No DOI is invented.

Verified sources:
- https://arxiv.org/html/1706.03762v7
- https://arxiv.org/abs/1706.03762
- https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- https://github.com/tensorflow/tensor2tensor (authors' code, Apache-2.0, archived)

## A. What is proposed?

**New architecture and attention formulation**, with empirical translation/parsing
evidence and a parallelization argument. It combines scaled dot-product attention,
multi-head attention, sinusoidal positions, post-normalized residual stacks,
pointwise ReLU networks and autoregressive decoding. It is not a new dataset,
loss family, optimizer, or mathematical proof of universal superiority.
Adam, label smoothing, encoder–decoder attention, and residual connections build on prior work.

Problem: recurrent transduction has sequential training computation and long paths
between positions. Hypothesis: attention-only stacks can improve translation
quality while permitting parallel training and shorter dependency paths.
Important distinction: teacher-forced training is parallel across target positions;
autoregressive inference still requires sequential token generation.

## B. Reproducibility classification

| Component | Classification | Evidence / boundary |
|---|---|---|
| Attention Eq. 1, multi-head projections | Reproducible | §3.2 dimensions and computation |
| Encoder/decoder post-norm topology | Reproducible | §3.1 and Figure 1 |
| ReLU FFN, tied scaled embeddings, sinusoidal PE | Reproducible | §3.3–3.5 |
| Adam and schedule Eq. 3 | Reproducible | §5.3 constants |
| Residual dropout and smoothing strength | Partially reproducible | §5.4 strength given, exact smoothing distribution not given |
| Beam search/checkpoint averaging | Partially reproducible | §6.1 settings given, search termination and checkpoint code details need interpretation |
| WMT training and exact BLEU | Partially reproducible | corpus recipe, tokenization/scoring, random seed and initialization under-specified |
| Exact historical bitwise run | Not reproducible from paper alone | original full state/environment unavailable |
| WSJ and 17M parsing corpora | Partially reproducible | access restrictions, semi-supervised construction/tuning missing |

Author-reported Table 2: base EN-DE 27.3 / EN-FR 38.1 BLEU;
big EN-DE 28.4 / EN-FR 41.8 BLEU. **§6.1 prose reports 41.0 EN-FR**.
The proceedings abstract page also carries different historical values (27.5 / 41.1).
These are version/source discrepancies, not implementation measurements.
Paper hardware: 8 NVIDIA P100s; base 100K steps (~12h), big 300K (~3.5 days).
No local run is evidence of these reported WMT results.

## C. ASSUMPTIONS & AMBIGUITIES REGISTER

| ID | Paper says / missing detail | Engineering interpretation and choice | Alternative | Reason and reproducibility impact |
|---|---|---|---|---|
| A1 | Post-residual LayerNorm; epsilon unspecified | PyTorch LayerNorm epsilon 1e-5 | Historical implementation epsilon | Explicit stable default; affects numerical fidelity |
| A2 | Projection matrices without attention biases; initialization absent | Bias-free Q/K/V/O, Xavier linear weights, normal embedding std d^-0.5 | Biased projections, historical initializer | Closest to equations; not bitwise reproduction |
| A3 | Smoothing epsilon=.1; support unspecified | True class 1-epsilon; distribute over non-pad non-true classes | Uniform over all classes (PyTorch default) | Padding is not a token target; changes training objective from other implementations |
| A4 | Three-way shared embedding matrix | Shared joint train-only vocabulary; tie encoder/decoder/output | Separate vocabularies and output-only tying | §3.4 directly represented; affects vocab construction |
| A5 | Residual dropout specified; attention dropout mentioned for parsing | Residual/embedding dropout plus configurable attention dropout (default 0) | Apply .1 attention dropout too | Avoid silently attributing extra regularization to translation paper |
| A6 | Beam 4, alpha .6; termination partly delegated to prior work | Bounded exhaustive beam expansion; retain completed beams; GNMT length normalization | Historical early stopping / caching | Transparent executable search; differs in speed/tie breaking |
| A7 | Padding/all-masked queries not specified | Boolean True=allowed; fully masked row gives zero attention | Reject all-masked row | Finite outputs/gradients; behavior outside paper's usual valid inputs |
| A8 | 37K BPE / 32K wordpieces, exact recipe not fixed | Local adapter uses whitespace, train-only vocabulary | Supply original BPE pipeline | Local data is an engineering validation path, not WMT reproduction |
| A9 | Last 5/20 checkpoints averaged | Utility averages compatible safetensors states; local trainer saves final state | Time-based historical averaging | Mechanism implemented, historical checkpoint cadence not reproduced |
| A10 | Full-scale batch ~25K source/target tokens | Local runs use fixed sentence batch, CPU default, optional CUDA | Length-bucketed token-budget multi-GPU batching | Bounded runnable lab; major training-scale deviation |

### Missing-information format

MISSING FROM PAPER: exact dataset filtering, scoring signature, seed, initialization,
LayerNorm epsilon and label-smoothing distribution.

ENGINEERING ASSUMPTION: A1–A10 above define all choices used by this lab.

WHY THIS ASSUMPTION: independently testable, stable, understandable execution.

IMPACT: equations/topology are validated; exact scores and historical trajectories are unverified.

ALTERNATIVE: reconstruct the historical Tensor2Tensor release and WMT preprocessing
with fixed corpus checksums and licensed corpus access before attempting claim reproduction.

## Research decomposition

Problem → sequential bottleneck; motivation → parallelism and distant dependencies;
question → can attention replace recurrence/convolution?; hypothesis → yes for transduction;
contributions → attention-only Transformer and translation/parsing evaluation;
theory → dot-product variance scaling and dependency path argument;
math → §3/5 equations; components → embeddings, PE, attention, FFN, Add & Norm;
training → shifted teacher forcing, Adam, dropout, smoothing;
inference → autoregressive beam search; data → WMT/WSJ;
experiments → translation/parsing; benchmarks → BLEU/F1 and estimated training FLOPs;
ablations → heads, key dimension, depth, width, dropout, smoothing, learned positions;
limitations → quadratic attention, sequential decoding, dataset-specific evidence;
conclusion → viable attention-only transduction.

## Implementation strategy (recorded before core implementation)

- Python/PyTorch research logic in `src/research_lab/papers/attention/`.
- Shared typed configs, dataset adapters, safe artifact store, benchmarks and registry.
- FastAPI transport; one bounded background worker prevents concurrent training RNG interference.
- React/TypeScript presentation; registry catalog, educational content and actual measured runs.
- JSON/YAML config and content; safetensors model storage; no arbitrary code or pickle uploads.
- Correctness first: numerical oracle, gradient/mask/padding invariants, then SDPA equivalence.
- Experiments: disjoint synthetic copy/reversal; copy/majority baselines; optional learned GRU;
  retrained positional/head/smoothing ablations with identical datasets and seeds.
- Metrics: free-running exact match/token accuracy and teacher-forced NLL/perplexity;
  BLEU only for uploaded parallel-text tasks, with SacreBLEU signature.
- Resource metrics: repeated CPU/CUDA inference timing, parameter bytes, process RSS and
  environment. Cost, VRAM and GPU metrics unavailable on CPU are not inferred.
- Deployment: one application container serving built UI and API, persistent runs volume;
  tests/type/lint/build/audit CI and manually triggered image publishing.

## Planned traceability

See `TRACEABILITY.md`; each substantial component has section/equation, source,
scientific test and experiment mapping. WMT claim validation remains NOT IMPLEMENTED.
