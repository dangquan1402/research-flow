---
title: "Experiment: Position Coupling vs Learned APE for Addition"
created: 2026-04-14
updated: 2026-04-14
source: experiment
confidence: medium
tags: [experiment, position-coupling, APE, positional-encoding, comparison]
---

# Position Coupling vs Learned APE

## Comparison Summary

| Metric | Learned APE | Position Coupling |
|---|---|---|
| Final ID accuracy | 99.95% | 77.05% |
| Convergence (50 epochs) | Yes | No |
| Final loss | 1.278 | 1.092 |
| Parameters | 2,782,464 | 2,866,560 |
| Training time | 1778s | 1781s |
| OOD 6-digit | 0.0% | 4.6% |
| OOD 7-digit | 0.0% | 0.8% |
| OOD 8-10 digit | 0.0% | ≤0.2% |

## Analysis

### Training Dynamics
- **APE converges much faster**: reaches 99.95% by epoch 15, maintains through epoch 50
- **PC learns more slowly**: steadily improving but only reaches 77% at epoch 50. Loss is lower (1.09 vs 1.28) suggesting the model is learning different representations that don't yet translate to exact-match accuracy
- **PC parameter count slightly higher** (+84k, 3%) due to larger position embedding table (PC_MAX_POS=256 vs max_seq_len=37)

### In-Distribution Performance
- APE dominates on ID accuracy: 99.95% vs 77%
- PC particularly struggles on 1–2 digit examples (20%, 29%) — likely because the random training offset [1, 100] creates disproportionate positional noise for very short sequences

### Out-of-Distribution Performance
- APE: hard cliff at training boundary (0% for all OOD lengths)
- PC: marginal OOD signal (4.6% at 6 digits), rapidly decaying
- **Comparison is confounded** by PC's lack of convergence — cannot fairly assess OOD capability of a model that hasn't mastered ID

## Verdict

**Inconclusive.** Position Coupling shows promise (non-zero OOD, lower loss trajectory) but 50 epochs is insufficient for convergence. The experiment should be repeated with 200+ epochs before drawing conclusions about length generalization capability.

## What Would Disprove This?

- If PC converges at 200 epochs but still shows 0% OOD: the approach doesn't work with reversed output format
- If PC never converges even at 500 epochs: the implementation may be incorrect (check position ID assignment logic)
