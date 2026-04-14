---
title: "Experiment: 3-Digit Multiplication Baseline with 2L/384D Transformer"
created: 2026-04-13
updated: 2026-04-13
source: experiments/results/mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json
confidence: high
verification: source
tags: [experiment, arithmetic, multiplication, M4]
---

## Summary

A 2-layer, 384-dimension transformer achieves **85.15% exact-match accuracy** on 3-digit multiplication with reversed output — significantly below the 99.9%+ achieved on addition and subtraction. Multiplication is fundamentally harder, with accuracy degrading sharply as output length increases.

## Configuration

| Parameter | Value |
|---|---|
| Operation | Multiplication (a * b) |
| Max digits | 3 |
| Architecture | 2L / 4H / 384D |
| FF dimension | 1536 |
| Parameters | 3,560,064 |
| Tokenizer | Reversed (LSB-first) |
| Balanced carry | No (uses balanced digit sampling) |
| Training samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 80 |
| Batch size | 256 |
| Training time | 606.9s (~10.1 min) on Apple M4 |

## Results

| Epoch | Accuracy | Notes |
|---|---|---|
| 5 | 41.45% | Learns 1-digit products quickly (99%) |
| 10 | 58.60% | 2-digit results at 94% |
| 15 | 67.95% | 3-digit at 97%, 4-digit at 84% |
| 20 | 71.20% | Plateau beginning |
| 30 | 75.25% | Slow improvement |
| 45 | 81.25% | Gradual climb |
| 60 | 83.10% | |
| 70 | 85.35% | Peak |
| 80 | 85.15% | Final — plateaued |

### Per-Digit-Count Accuracy (Final, by result digit count)

| Result digits | Accuracy |
|---|---|
| 1 | 100% |
| 2 | 100% |
| 3 | 100% |
| 4 | 96% |
| 5 | 66% |
| 6 | 32% |

## Key Observations

1. **Multiplication is fundamentally harder**: 85% vs 99.9% for add/sub. The model cannot fully learn the multi-step carry propagation required for long products.
2. **Sharp accuracy cliff by output length**: 100% for 1-3 digit results, drops to 96% for 4-digit, 66% for 5-digit, and 32% for 6-digit results. This suggests the 2-layer model lacks the depth to chain the partial-product accumulations needed for longer outputs.
3. **Slow convergence**: Still improving at epoch 80 (unlike add/sub which converge by epoch 20). More epochs might help marginally but the model appears capacity-limited.
4. **Reversal still helps**: Even at 85%, reversed output is critical — without it, multiplication accuracy would likely be much lower due to carry alignment issues.

## Comparison to Addition/Subtraction

| Metric | Addition (5d, 2L/384D) | Subtraction (5d, 2L/384D) | Multiplication (3d, 2L/384D) |
|---|---|---|---|
| Final accuracy | 99.95% | 99.90% | 85.15% |
| Max operand digits | 5 | 5 | 3 |
| Max result digits | 6 | 5 | 6 |
| Convergence | ~20 epochs | ~20 epochs | Not converged at 80 |
| Training time | ~390s | ~460s | ~607s |

## What Would Disprove This?

- If a deeper model (4L) with the same parameter budget achieves >95% on multiplication, it would confirm that depth (not width) is the bottleneck for multi-step arithmetic.
- If scratchpad/chain-of-thought output format enables >95% accuracy even with a shallow model, it would suggest the limitation is output format, not model capacity.

## Citation

[mul_3d_reversed_2L4H384D_mul_baseline_2L384D] experiments/results/mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json
