---
title: "Experiment: Scratchpad Multiplication with 4L/256D — 100% Accuracy"
created: 2026-04-14
updated: 2026-04-14
source: experiments/results/mul_3d_reversed_4L4H256D_scratchpad_4L256D.json
confidence: high
verification: source
tags: [experiment, arithmetic, multiplication, scratchpad, M4]
related: [experiment-multiplication-4L256D, scratchpad-format-research]
staleness_days: 0
---

## Summary

Adding scratchpad (aligned reversed partial products) to the 4L/256D transformer pushes 3-digit multiplication from **94.90% to 100.00%** — a **+5.1 percentage point improvement** that fully solves multiplication. 6-digit results go from 60% to 100%.

## Configuration

| Parameter | Value |
|---|---|
| Operation | Multiplication (a * b) |
| Max digits | 3 |
| Architecture | 4L / 4H / 256D |
| FF dimension | 1024 |
| Parameters | 3,165,952 |
| Tokenizer | Reversed (LSB-first) + scratchpad |
| Scratchpad format | Aligned reversed partial products with `\|` separator |
| Max sequence length | 34 tokens |
| Training samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 80 |
| Batch size | 256 |
| Training time | 2340.2s (~39 min) on Apple M4 |

## Results

| Epoch | Accuracy | 6-digit acc | Notes |
|---|---|---|---|
| 5 | 87.65% | 28% | Already near non-scratchpad final (94.9%) |
| 10 | 97.80% | 90% | Massive jump — scratchpad enables rapid learning |
| 15 | 99.05% | 91% | Passes target (>99%) by epoch 15 |
| 25 | 99.35% | 95% | |
| 40 | **99.95%** | **99%** | Near-perfect |
| 55 | 99.80% | 99% | |
| 70 | **100.00%** | **100%** | Perfect accuracy |
| 80 | **100.00%** | **100%** | Confirmed |

## Comparison: Scratchpad vs No-Scratchpad (4L/256D)

| Metric | No Scratchpad | With Scratchpad | Delta |
|---|---|---|---|
| Final accuracy | 94.90% | **100.00%** | **+5.10pp** |
| 5-digit result acc | 93% | 100% | +7pp |
| 6-digit result acc | 60% | **100%** | **+40pp** |
| Epoch to 99% | Never | Epoch 15 | -- |
| Training time | 664s | 2340s | 3.5x longer |
| Sequence length | ~15 tokens | ~34 tokens | 2.3x |

## Key Observations

1. **Scratchpad fully solves multiplication** for 3-digit operands with 4L/256D.
2. **6-digit results go from 60% to 100%** — the accuracy cliff that plagued non-scratchpad is completely eliminated.
3. **99% accuracy by epoch 15** — faster convergence to target than non-scratchpad reaches even after 80 epochs.
4. **Training time increases 3.5x** (664s → 2340s) due to longer sequences, but this is acceptable for the quality improvement.
5. **Some oscillation** in accuracy (dips at epochs 35, 45) likely due to the model exploring different scratchpad generation strategies during training.

## What Would Disprove This?

- If the same accuracy could be achieved without scratchpad by simply training longer (>200 epochs), the scratchpad benefit would be about convergence speed, not capacity.
- If scratchpad failed on 4+ digit operands, the format might not generalize beyond 3-digit multiplication.

## Citation

[mul_3d_reversed_4L4H256D_scratchpad_4L256D] experiments/results/mul_3d_reversed_4L4H256D_scratchpad_4L256D.json
