---
title: "Experimental Finding: Looped Transformer Achieves 99.4% on 3-Digit Mul+Scratchpad with 456K Params"
created: 2026-04-14
updated: 2026-04-14
source: experiment-looped-mul-scratchpad
confidence: high
verification: source
tags: [experiment, arithmetic, multiplication, looped-transformer, scratchpad, weight-sharing, parameter-efficiency]
related: [looped-transformers-parameter-efficiency, experiment-looped-192D-addition, experiment-scratchpad-2L384D, scratchpad-chain-of-thought-arithmetic]
staleness_days: 0
---

## Insight

A looped transformer (1Lx4, 4H, 192D) achieves **99.4% exact accuracy** on 3-digit multiplication with scratchpad using only **456,384 parameters** — 87.2% fewer params than the 2L/384D baseline (3.56M params, 100%). Looped transformers with scratchpad are a highly parameter-efficient approach for multiplication.

## Evidence

Experimental results from `experiments/train.py` with `--architecture looped --scratchpad`:
- Config: 1L/4H/192D, n_loops=4, 50K train, 2K test, 80 epochs, batch_size=256
- Tokenizer: reversed, balanced carry sampling, scratchpad enabled
- Final accuracy: 99.4% (vs baseline 100%)
- Final loss: 0.605
- Total training time: 7340s (~122 min, with heavy CPU contention)
- Result file: `mul_3d_reversed_1L4H192D_looped_mul_scratchpad.json`

### Training Dynamics
- Fast start: 39.6% at epoch 5, 77.7% at epoch 10
- Rapid convergence: 95.4% at epoch 25
- Some oscillation: accuracy dips to 89.2% at epoch 35, then recovers to 95.9% at epoch 40
- Stabilizes: 98.9% at epoch 55, 99.4% at epoch 80

### Per-Digit Accuracy (epoch 80)
| Digits | Accuracy |
|--------|----------|
| 1      | 100%     |
| 2      | 100%     |
| 3      | 100%     |
| 4      | 100%     |
| 5      | 99.1%    |
| 6 (carry) | 95.3% |

### Comparison to Addition
- Mul+scratchpad learns faster initially (39.6% at epoch 5 vs 2.1% for addition)
- Likely because scratchpad breaks multiplication into addition sub-steps the looped architecture handles well
- The looping mechanism naturally aligns with the iterative partial-product accumulation in scratchpad multiplication

## Counter-argument (FUNGI)

This could be disproved if: (1) the accuracy oscillation (89% at epoch 35) indicates training instability that could worsen with larger digit counts, (2) scratchpad hides the difficulty — the model may just be learning addition within the scratchpad steps rather than true multiplication.

## Implications

The looped transformer is especially well-suited for scratchpad multiplication because:
1. Weight sharing maps naturally to the repeated addition steps in scratchpad format
2. Input injection provides consistent access to the problem throughout processing
3. 456K params is sufficient for near-perfect 3-digit multiplication — a 7.8x efficiency gain over baseline
