---
title: "Experimental Finding: SwiGLU Converges Faster Than GELU at Equal Parameter Count"
created: 2026-04-14
updated: 2026-04-14
source: experiment-swiglu-fair-comparison
confidence: high
verification: source
tags: [experiment, activation-function, swiglu, gelu, addition, mixed-ops, fair-comparison]
related: [minimum-viable-math-transformer, experiment-baseline-addition-reversed, depth-vs-width-arithmetic]
staleness_days: 0
---

## Insight

At identical parameter counts (3,562,368), SwiGLU reaches 100% accuracy 5 epochs earlier than GELU on 5-digit addition (epoch 15 vs epoch 20), while also training 7% faster in wall-clock time. Both activations converge to equivalent final accuracy (~99.8-100%). A prior comparison was invalid due to a double-reduction bug that gave SwiGLU only 2.77M params vs GELU's 3.56M.

## What Would Disprove This?

- SwiGLU's advantage disappearing on harder tasks (multiplication, mixed-ops) where the extra gating may not help
- The convergence advantage being seed-dependent rather than systematic
- Larger models showing no difference, suggesting this is a small-model artifact

## Experimental Setup

### Experiment 1: GELU Baseline (5-digit addition)
- **Config:** 2L/4H/384D, ff_dim=1536, activation=gelu
- **FFN structure:** 2 matrices of (384, 1536) per layer
- **Params:** 3,562,368
- **Training:** 50 epochs, batch_size=256, lr=0.001, reversed tokenizer, balanced carry

### Experiment 2: SwiGLU Fair (5-digit addition)
- **Config:** 2L/4H/384D, ff_dim=1024, activation=swiglu
- **FFN structure:** 3 matrices of (384, 1024) per layer (gate, up, down)
- **Params:** 3,562,368 (exact match)
- **Training:** identical to GELU

### Experiment 3: SwiGLU Mixed-Ops
- **Config:** 2L/4H/384D, ff_dim=1024, activation=swiglu, scratchpad=true
- **Task:** mixed (add, sub, mul)
- **Params:** 3,581,184 (slightly more due to longer max_seq_len embedding)
- **Training:** 50 epochs, batch_size=256

## Results

### Addition (Fair Comparison — Identical 3,562,368 Params)

| Metric | GELU (ff_dim=1536) | SwiGLU (ff_dim=1024) |
|--------|-------------------|---------------------|
| Epoch to 95% | 10 | 10 |
| Epoch to 100% | 20 | **15** |
| Final accuracy | 99.95% | 99.75% |
| Training time | 2031s | **1882s** (-7%) |

### Epoch-by-Epoch Accuracy

| Epoch | GELU | SwiGLU |
|-------|------|--------|
| 5 | 26.50% | **84.05%** |
| 10 | 97.85% | 95.10% |
| 15 | 99.85% | **100.00%** |
| 20 | 100.00% | 99.95% |
| 25 | 98.40% | 99.90% |
| 30 | 100.00% | 100.00% |
| 35 | 99.95% | 100.00% |
| 40 | 99.75% | 100.00% |
| 45 | 99.95% | 99.75% |
| 50 | 99.95% | 99.75% |

### Mixed-Ops with Scratchpad (SwiGLU only, no param-matched GELU baseline)

- Final accuracy: 89.10% at epoch 50
- Digit breakdown at epoch 50: 1d:100%, 2d:100%, 3d:100%, 4d:97%, 5d:96%, 6d:82%, 7d:56%, 8d:50%, 9d:16%, 10d:0%
- Note: prior GELU mixed-ops used ff_dim=1024 (2.78M params) over 80 epochs → 98.75%, so not directly comparable

## Key Observations

1. **SwiGLU learns early representations dramatically faster**: 84% vs 27% at epoch 5, suggesting the gating mechanism helps the model discover digit-level patterns sooner
2. **Convergence speed advantage**: SwiGLU hits 100% at epoch 15, GELU at epoch 20 — a 25% reduction in epochs to convergence
3. **Wall-clock efficiency**: SwiGLU is 7% faster despite having 3 matrices per FFN layer (vs 2 for GELU), likely because the smaller matrices (384×1024 vs 384×1536) are more cache-friendly
4. **Both plateau similarly**: Final accuracy is essentially equivalent (~99.8-100%), so SwiGLU's advantage is speed, not ceiling
5. **Mixed-ops**: SwiGLU reaches 89% in 50 epochs on mixed-ops with scratchpad — reasonable but needs more epochs or architectural changes for full convergence

## Bug Context

The prior SwiGLU comparison (before commit `9af7d68`) had a double-reduction bug: the trainer auto-reduced ff_dim by 2/3 for SwiGLU, on top of the user already passing a reduced value. This gave SwiGLU only 682 ff_dim (2,774,400 params) vs GELU's 1536 ff_dim (3,562,368 params) — a 22% parameter disadvantage that invalidated the comparison.

## Evidence

- GELU result: `experiments/results/add_5d_reversed_2L4H384D_act_gelu.json`
- SwiGLU result: `experiments/results/add_5d_reversed_2L4H384D_swiglu_fair_swiglu.json`
- Mixed-ops result: `experiments/results/mixed_5d_reversed_2L4H384D_swiglu_fair_mixed.json`
