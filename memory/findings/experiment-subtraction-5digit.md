---
title: "Experiment: 5-Digit Subtraction with 2L/384D Transformer"
created: 2026-04-13
updated: 2026-04-13
source: experiments/results/sub_5d_reversed_2L4H384D_sub_2L384D.json
confidence: high
verification: source
tags: [experiment, arithmetic, subtraction, M4]
---

## Summary

A 2-layer, 384-dimension transformer achieves **99.90% exact-match accuracy** on 5-digit subtraction with reversed (LSB-first) output, matching the near-perfect performance seen on addition with the same architecture.

## Configuration

| Parameter | Value |
|---|---|
| Operation | Subtraction (a - b, a >= b) |
| Max digits | 5 |
| Architecture | 2L / 4H / 384D |
| FF dimension | 1536 |
| Parameters | 3,561,216 |
| Tokenizer | Reversed (LSB-first) |
| Balanced carry | Yes |
| Training samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 50 |
| Batch size | 256 |
| Training time | 460.5s (~7.7 min) on Apple M4 |

## Results

| Epoch | Accuracy | Notes |
|---|---|---|
| 5 | 3.50% | Still learning digit patterns |
| 10 | 59.15% | Rapid learning phase begins |
| 15 | 96.65% | Near-convergence |
| 20 | 99.45% | Approaching ceiling |
| 25 | 99.75% | All digit counts at 100% |
| 35 | 99.80% | Stable at ceiling |
| 40 | 99.90% | Peak accuracy |
| 50 | 99.90% | Final — stable |

### Per-Digit Accuracy (Final)

| Digits | Accuracy |
|---|---|
| 1 | 100% |
| 2 | 100% |
| 3 | 100% |
| 4 | 100% |
| 5 | 100% |

## Key Observations

1. **Subtraction matches addition difficulty**: The 2L/384D model achieves effectively the same accuracy on subtraction as on addition, confirming that both operations have similar algorithmic complexity for digit-level reversed-output transformers.
2. **Rapid convergence**: Jumps from 3.5% to 59% between epochs 5-10, then to 96.7% by epoch 15. The model learns subtraction's borrow-propagation pattern quickly.
3. **Balanced carry sampling helps**: Even though subtraction uses borrows (not carries), the balanced sampling strategy (which generates diverse borrow chain lengths) contributes to robust training.
4. **No architecture scaling needed**: The same 2L/384D architecture that solved addition also solves subtraction — no need to increase depth or width for this operation.

## Comparison to Addition Baseline

| Metric | Addition (2L/384D) | Subtraction (2L/384D) |
|---|---|---|
| Final accuracy | 99.95% | 99.90% |
| Convergence epoch | ~20 | ~20 |
| Training time | ~390s | ~460s |
| Parameters | 3,561,216 | 3,561,216 |

## What Would Disprove This?

- If subtraction with larger digit counts (e.g., 8-10 digits) showed significantly worse accuracy than addition, it would suggest borrow propagation is harder than carry propagation at scale.
- If removing balanced carry sampling caused a larger accuracy drop for subtraction than addition, it would indicate subtraction is more sensitive to training distribution.

## Citation

[sub_5d_reversed_2L4H384D_sub_2L384D] experiments/results/sub_5d_reversed_2L4H384D_sub_2L384D.json
