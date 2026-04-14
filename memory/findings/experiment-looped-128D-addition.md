---
title: "Experimental Finding: Looped Transformer 1Lx4/128D Fails on 5-Digit Addition (37.7%) — Capacity Floor"
created: 2026-04-14
updated: 2026-04-14
source: experiment-looped-1Lx4-128D
confidence: high
verification: source
tags: [experiment, arithmetic, addition, looped-transformer, parameter-efficiency, negative-result]
related: [looped-transformers-parameter-efficiency, experiment-looped-192D-addition, depth-vs-width-arithmetic, minimum-viable-math-transformer]
staleness_days: 0
---

## Insight

A looped transformer with 1 layer block looped 4 times (1Lx4), 4 heads, 128 dimensions achieves only **37.7% exact accuracy** on 5-digit addition with **204,160 parameters**. This establishes a capacity floor: 128D is insufficient for the looped architecture to learn 5-digit addition, even with 80 epochs of training. The model was still slowly improving but showed no sign of a phase transition.

## Evidence

Experimental results from `experiments/train.py` with `--architecture looped`:
- Config: 1L/4H/128D, n_loops=4, 50K train, 2K test, 80 epochs, batch_size=256
- Tokenizer: reversed, balanced carry sampling
- Final accuracy: 37.7% (vs baseline 99.9%, vs 192D looped 99.2%)
- Final loss: 1.4763
- Total training time: 3071s (~51 min, with heavy CPU contention)
- Result file: `add_5d_reversed_1L4H128D_looped_1Lx4_128D.json`

### Training Dynamics
- Very slow learning: 2.4% at epoch 5, 20.4% at epoch 10
- Plateau region: 21-27% from epochs 15-35 (minimal improvement)
- Gradual climb: 28.5% at epoch 45, 33.2% at epoch 65, 37.7% at epoch 80
- No phase transition observed (unlike the 192D variant which had a sharp jump at epoch 25-40)

### Per-Digit Accuracy (epoch 80)
| Digits | Accuracy |
|--------|----------|
| 1      | 100%     |
| 2      | 99.4%    |
| 3      | 94.3%    |
| 4      | 41.7%    |
| 5      | 12.1%    |
| 6 (carry) | 9.7%  |

Accuracy drops sharply for 4+ digit inputs — the model learns easier cases but cannot handle carry propagation across many digits.

## Counter-argument (FUNGI)

This could be disproved if: (1) significantly more training (200+ epochs) triggers a delayed phase transition, or (2) a different learning rate schedule could help the 128D model find the right representation. However, the loss curve flattening at ~1.48 suggests a capacity bottleneck rather than an optimization issue.

## Implications

There is a sharp capacity threshold between 128D and 192D for looped transformers on 5-digit addition. The 192D model (453K params) achieves 99.2% while the 128D model (204K params) only reaches 37.7%. This ~2x parameter difference results in a ~60pp accuracy gap, suggesting the minimum viable dimension for looped arithmetic transformers is somewhere in the 128-192D range.
