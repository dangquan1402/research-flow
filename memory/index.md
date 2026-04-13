# Memory Index

> Auto-updated after every ingest, analyze, and synthesize operation.
> Each entry: `- [Title](path) — one-line summary | confidence | updated`

## Entities

- [Teaching Arithmetic to Small Transformers](entities/paper-teaching-arithmetic-small-transformers.md) — Lee et al. ICLR 2024, core reference for small arithmetic transformers | 2026-04-13
- [Transformers Can Do Arithmetic with Right Embeddings](entities/paper-transformers-arithmetic-right-embeddings.md) — McLeish 2024, Abacus Embeddings for length generalization | 2026-04-13
- [Position Coupling](entities/paper-position-coupling.md) — Cho 2024, positional encoding enabling 6.67x length generalization | 2026-04-13
- [Grokking: Generalization Beyond Overfitting](entities/paper-grokking-power-2022.md) — Power et al. 2022, sudden generalization after overfitting | 2026-04-13
- [Progress Measures for Grokking](entities/paper-progress-measures-grokking-nanda.md) — Nanda et al. 2023, mechanistic interpretability of grokking | 2026-04-13
- [Length Generalization in Arithmetic](entities/paper-length-generalization-arithmetic.md) — Jelassi et al. 2023, techniques for N→N+k generalization | 2026-04-13
- [Tokenization Counts](entities/paper-tokenization-counts.md) — Stolfo et al. 2024, impact of tokenization on arithmetic | 2026-04-13
- [Reverse That Number / LEFT](entities/paper-reverse-that-number-LEFT.md) — Proves reversed output reduces complexity from exponential to linear | 2026-04-13
- [The Depth Delusion](entities/depth-delusion-paper.md) — 2026 paper showing width dominates depth for algorithmic tasks | 2026-04-13
- [Understanding Addition in Transformers](entities/understanding-addition-transformers.md) — Nikankin 2024, mechanistic analysis of addition | 2026-04-13
- [Andrej Karpathy](entities/andrej-karpathy.md) — nanoGPT/minGPT creator, transformer fundamentals reference | 2026-04-13
- [Rodrigo Nogueira](entities/researcher-nogueira.md) — Pioneer of position tokens for arithmetic | 2026-04-13

## Findings

### Tokenization (Hypothesis A — GH-2)
- [Digit-Level Tokenization](findings/digit-level-tokenization.md) — Each digit must be its own token; non-negotiable prerequisite | high | 2026-04-13
- [Reversed Digit Order](findings/reversed-digit-order.md) — LSB-first output aligns with carry propagation, 4-10x sample efficiency | high | 2026-04-13
- [BPE Hurts Arithmetic](findings/bpe-hurts-arithmetic.md) — BPE misaligns place values, drops accuracy to 8.25% on length mismatch | high | 2026-04-13
- [Index Hints & Position Markers](findings/index-hints-position-markers.md) — Evolution from index hints to Abacus Embeddings to Position Coupling | high | 2026-04-13
- [Scratchpad / Chain-of-Thought](findings/scratchpad-chain-of-thought.md) — Essential for multiplication, optional for add/sub | medium | 2026-04-13
- [Fixed-Width Padding Superseded](findings/fixed-width-padding-superseded.md) — Useful baseline but superseded by position-aware embeddings | medium | 2026-04-13

### Architecture (Hypothesis B — GH-3)
- [Depth vs Width](findings/depth-vs-width-arithmetic.md) — Width dominates; 2-4 layers optimal; deeper causes gradient starvation | high | 2026-04-13
- [Minimum Viable Math Transformer](findings/minimum-viable-math-transformer.md) — Conservative: 4L/4H/384d/~10M params; aggressive: 2L/4H/256d/~2-3M | high | 2026-04-13
- [Positional Encoding](findings/positional-encoding-arithmetic.md) — Standard PEs fail at length gen; Position Coupling/Abacus excel; NoPE surprisingly good | high | 2026-04-13
- [Attention Heads & Carry Operations](findings/attention-heads-carry-operations.md) — 3 heads minimum: base addition, carry detect, carry cascade | medium | 2026-04-13
- [Data Formatting Over Architecture](findings/data-formatting-over-architecture.md) — Reversed output is bigger win than any architecture change (85%→100%) | high | 2026-04-13
- [Looped Transformers](findings/looped-transformers-parameter-efficiency.md) — 2 layers x 8 loops = 16 effective layers at 2-layer cost | medium | 2026-04-13
- [Normalization](findings/normalization-small-arithmetic-models.md) — Pre-norm (RMSNorm) essential; eliminates warmup need | medium | 2026-04-13

### Training (Hypothesis C — GH-4)
- [Curriculum Learning](findings/curriculum-learning-arithmetic.md) — Helps sample efficiency but data format matters more | medium | 2026-04-13
- [Data Format Impact](findings/data-format-impact.md) — Single most important decision; reverse/scratchpad dramatically outperform plain | high | 2026-04-13
- [Grokking Arithmetic](findings/grokking-arithmetic.md) — Real but fragile; weight decay is key lever; three-phase mechanism | high | 2026-04-13
- [Length Generalization Techniques](findings/length-generalization-techniques.md) — Position coupling (30-digit→200-digit) + balanced carry sampling | high | 2026-04-13
- [Optimizer & Schedule](findings/optimizer-schedule-arithmetic.md) — AdamW, lr=1e-3, beta2=0.99, wd=0.5, batch 256, cosine + 100-step warmup | high | 2026-04-13
- [Scratchpad for Arithmetic](findings/scratchpad-chain-of-thought-arithmetic.md) — Dramatic accuracy boost but 3-10x sequence length cost | medium | 2026-04-13
- [Balanced Sampling Strategy](findings/balanced-sampling-strategy.md) — Balanced carry sampling critical; uniform random severely skews data | high | 2026-04-13
- [Data Augmentation](findings/data-augmentation-arithmetic.md) — Commutativity, identity ops, zero-padding: low-cost high-value | medium | 2026-04-13

## Themes

- [Data Format Supremacy](themes/data-format-supremacy.md) — Formatting decisions (reversed output, digit-level tokenization) outweigh all architecture choices | high | 2026-04-13
- [Shallow and Wide Architecture](themes/shallow-wide-architecture.md) — 2-4 layers with 4+ heads is optimal; width grows 2.8x faster than depth | high | 2026-04-13
- [Position Encoding Unlocks Generalization](themes/position-encoding-unlocks-generalization.md) — Standard PEs fail; Abacus/Position Coupling enable 6-7x length generalization | high | 2026-04-13
- [Training Data Distribution](themes/training-data-distribution.md) — Balanced carry sampling and augmentation matter as much as dataset volume | high | 2026-04-13
- [Regularization Drives Generalization](themes/regularization-drives-generalization.md) — Weight decay 0.1-1.0 controls memorization→algorithm transition (grokking) | high | 2026-04-13

### Experimental Results (2026-04-13)
- [Baseline Addition Reversed](findings/experiment-baseline-addition-reversed.md) — 4L/4H/256D achieves 99.9% on 5-digit addition with reversed output | high | 2026-04-13
- [Tokenization Comparison](findings/experiment-tokenization-comparison.md) — Reversed +2% accuracy and 2x learning speed over plain output | high | 2026-04-13
- [Depth vs Width Ablation](findings/experiment-depth-vs-width-ablation.md) — 2L/4H/384D is sweet spot; all configs >99.5% on addition | high | 2026-04-13

## Open Questions

- [Combining Techniques Interaction Effects](open-questions/combining-techniques-interaction-effects.md) — How do reversed output + position coupling + scratchpad interact?
- [Grokking on Integer vs Modular Arithmetic](open-questions/grokking-on-integer-vs-modular-arithmetic.md) — Does grokking apply to unbounded integers?
- [Length Generalization Necessity](open-questions/length-generalization-necessity.md) — Is N→N+k generalization needed for our scope?
- [Mixed Operation Architecture](open-questions/mixed-operation-architecture.md) — Minimum arch for add+sub+mult simultaneously?
- [Multiplication Length Generalization](open-questions/multiplication-length-generalization.md) — Remains unsolved in literature
- [Multiplication Scaling Requirements](open-questions/multiplication-scaling-requirements.md) — How does mult difficulty scale with digit count?
- [Optimal Multiplication Scratchpad Format](open-questions/optimal-multiplication-scratchpad-format.md) — Best intermediate step format?
- [Optimal Weight Decay Tradeoff](open-questions/optimal-weight-decay-tradeoff.md) — Needs empirical tuning for our setup
- [Position Coupling vs Abacus Head-to-Head](open-questions/position-coupling-vs-abacus-head-to-head.md) — No published comparison exists
- [Scratchpad vs Reverse Practical Tradeoff](open-questions/scratchpad-vs-reverse-practical-tradeoff.md) — When does scratchpad justify its cost?
