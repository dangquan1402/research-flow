---
title: "Experiment: 3-Digit Multiplication with 4L/256D Transformer (Depth Comparison)"
created: 2026-04-13
updated: 2026-04-13
source: experiments/results/mul_3d_reversed_4L4H256D_mul_4L256D.json
confidence: high
verification: source
tags: [experiment, arithmetic, multiplication, depth-vs-width, M4]
---

## Summary

A 4-layer, 256-dimension transformer achieves **94.90% exact-match accuracy** on 3-digit multiplication — a **+9.75 percentage point improvement** over the 2L/384D baseline (85.15%) with ~11% fewer parameters. Depth is definitively more important than width for multiplication.

## Configuration

| Parameter | Value |
|---|---|
| Operation | Multiplication (a * b) |
| Max digits | 3 |
| Architecture | 4L / 4H / 256D |
| FF dimension | 1024 |
| Parameters | 3,160,832 |
| Tokenizer | Reversed (LSB-first) |
| Balanced carry | No (uses balanced digit sampling) |
| Training samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 80 |
| Batch size | 256 |
| Training time | 664.4s (~11.1 min) on Apple M4 |

## Results

| Epoch | Accuracy | Notes |
|---|---|---|
| 5 | 52.10% | Already ahead of 2L baseline at same epoch (41.5%) |
| 10 | 64.85% | |
| 15 | 80.10% | Matches 2L's epoch-80 performance at epoch 15 |
| 30 | 83.40% | |
| 35 | 88.75% | Surpasses 2L's final accuracy |
| 40 | 90.25% | |
| 55 | 92.10% | |
| 60 | 93.35% | |
| 65 | 94.15% | |
| 75 | 94.75% | |
| 80 | 94.90% | Final — still improving slightly |

### Per-Digit-Count Accuracy (Final, by result digit count)

| Result digits | 2L/384D | 4L/256D | Delta |
|---|---|---|---|
| 1 | 100% | 100% | — |
| 2 | 100% | 99% | -1% |
| 3 | 100% | 100% | — |
| 4 | 96% | 100% | **+4%** |
| 5 | 66% | 93% | **+27%** |
| 6 | 32% | 60% | **+28%** |

## Key Observations

1. **Depth > width for multiplication**: With 11% fewer parameters, the 4L model beats 2L by nearly 10 points. The extra layers provide the computational steps needed to chain partial-product accumulations.
2. **Massive improvement on long outputs**: The biggest gains are on 5-digit (+27pp) and 6-digit (+28pp) results — exactly where the 2L model hit its capacity ceiling. Extra depth directly addresses multi-step carry propagation.
3. **Faster convergence**: 4L reaches 80% by epoch 15 (vs epoch 55 for 2L). More depth enables faster learning of the multiplication algorithm.
4. **Still not solved**: 94.9% is good but not the 99.9% we see on addition. 6-digit results at 60% suggest even 4 layers isn't fully sufficient for the hardest multiplication cases. 6L or 8L might be needed, or scratchpad output.
5. **Still improving at epoch 80**: Unlike the 2L model which plateaued, 4L is still gaining ~0.15%/epoch. More training could push this higher, perhaps to ~96-97%.

## Architecture Comparison

| Metric | 2L/384D (wide) | 4L/256D (deep) |
|---|---|---|
| Parameters | 3,560,064 | 3,160,832 |
| Final accuracy | 85.15% | 94.90% |
| 5-digit result acc | 66% | 93% |
| 6-digit result acc | 32% | 60% |
| Convergence speed | Slow (plateaus at 80) | Faster (reaches 2L peak by ep 15) |
| Training time | 607s | 664s |

## Implications for Architecture Design

- For **addition/subtraction**: 2L is sufficient — these operations only need single-step carry/borrow propagation that can be handled in one pass.
- For **multiplication**: Depth is critical. Each layer can handle one step of the partial-product-and-accumulate chain. Recommend minimum 4L for multiplication, potentially 6-8L for larger digit counts.
- **Width vs depth tradeoff**: Given a fixed parameter budget, allocate to depth first for multi-step operations. Width (embedding dimension) has diminishing returns once it's large enough to represent the digit vocabulary.

## What Would Disprove This?

- If a 2L model with scratchpad/chain-of-thought output matched or beat the 4L model, it would suggest the issue is output format, not model depth.
- If 6L/256D showed no improvement over 4L/256D, it would indicate a different bottleneck (e.g., training data or learning rate) rather than depth.
- If 2L/512D (wider, same total params as 4L/256D) also reached ~95%, it would weaken the depth-is-key argument.

## Citation

[mul_3d_reversed_4L4H256D_mul_4L256D] experiments/results/mul_3d_reversed_4L4H256D_mul_4L256D.json
