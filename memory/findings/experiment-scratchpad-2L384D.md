---
title: "Experiment: Scratchpad Multiplication with 2L/384D — 100% Accuracy, Shallow Now Viable"
created: 2026-04-14
updated: 2026-04-14
source: experiments/results/mul_3d_reversed_2L4H384D_scratchpad_2L384D.json
confidence: high
verification: source
tags: [experiment, arithmetic, multiplication, scratchpad, shallow-wide, M4]
related: [experiment-scratchpad-4L256D, scratchpad-format-research, experiment-multiplication-baseline]
staleness_days: 0
---

## Summary

Adding scratchpad to the shallow-wide 2L/384D transformer pushes 3-digit multiplication from **85.15% to 100.00%** — a **+14.85 percentage point improvement**. This is the key result: **scratchpad makes the shallow architecture fully viable for multiplication**, eliminating the depth bottleneck that previously required 4 layers.

## Configuration

| Parameter | Value |
|---|---|
| Operation | Multiplication (a * b) |
| Max digits | 3 |
| Architecture | 2L / 4H / 384D |
| FF dimension | 1536 |
| Parameters | 3,567,744 |
| Tokenizer | Reversed (LSB-first) + scratchpad |
| Scratchpad format | Aligned reversed partial products with `\|` separator |
| Max sequence length | 34 tokens |
| Training samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 80 |
| Batch size | 256 |
| Training time | 1970.1s (~33 min) on Apple M4 |

## Results

| Epoch | Accuracy | 6-digit acc | Notes |
|---|---|---|---|
| 5 | 74.00% | 5% | Slow start |
| 10 | 93.10% | 86% | Rapid convergence |
| 20 | **99.75%** | **98%** | Near-perfect |
| 30 | 99.85% | 99% | |
| 35 | **100.00%** | **100%** | First perfect score |
| 40 | **100.00%** | **100%** | Confirmed |
| 65 | **100.00%** | **100%** | Stable |
| 75 | **100.00%** | **100%** | Stable |
| 80 | **100.00%** | **100%** | Final — confirmed |

## Comparison: All 2L/384D Multiplication Results

| Metric | No Scratchpad | With Scratchpad | Delta |
|---|---|---|---|
| Final accuracy | 85.15% | **100.00%** | **+14.85pp** |
| 5-digit result acc | 66% | 100% | **+34pp** |
| 6-digit result acc | 32% | **100%** | **+68pp** |
| Epoch to 99% | Never | Epoch 20 | -- |
| Training time | 607s | 1970s | 3.2x longer |

## Comparison: 2L/384D Scratchpad vs 4L/256D Scratchpad

| Metric | 4L/256D + Scratch | 2L/384D + Scratch |
|---|---|---|
| Parameters | 3,165,952 | 3,567,744 |
| Final accuracy | 100.00% | 100.00% |
| First 100% epoch | Epoch 70 | **Epoch 35** |
| First 99% epoch | Epoch 15 | Epoch 20 |
| Training time | 2340s | **1970s** |

## Key Observations

1. **Scratchpad eliminates the depth bottleneck**: 2L/384D was 9.75pp behind 4L/256D without scratchpad. With scratchpad, both achieve 100% — the 2L model is no longer disadvantaged.
2. **2L reaches 100% faster** (epoch 35 vs epoch 70) — the shallow-wide architecture is actually better with scratchpad because width helps generate multiple tokens per step.
3. **6-digit results: 32% → 100%** — the most dramatic improvement of any experiment in this research.
4. **Training is 3.2x longer** than without scratchpad, but faster than 4L+scratchpad (1970s vs 2340s).
5. **Unified architecture now possible**: 2L/384D can handle add, sub, AND mul at >99.9%, eliminating the need for operation-specific architectures.

## Architecture Implication

**This result changes the recommendation.** Previously, we recommended 4L/256D for multiplication because depth was critical. With scratchpad, **2L/384D is the universal architecture** — it handles all three operations at >=99.9% accuracy. No need for operation-specific configurations.

## What Would Disprove This?

- If scratchpad failed on 4+ digit multiplication with 2L, depth might still be needed at scale.
- If mixed-operation training (add+sub+mul with scratchpad) caused interference, the unified architecture claim would need revision.
- If inference latency from scratchpad tokens is unacceptable, the 4L non-scratchpad approach might still be preferred.

## Citation

[mul_3d_reversed_2L4H384D_scratchpad_2L384D] experiments/results/mul_3d_reversed_2L4H384D_scratchpad_2L384D.json
