---
title: "Experimental Finding: Looped Transformer 1Lx4/192D Achieves 99.2% on 5-Digit Addition with 87% Fewer Params"
created: 2026-04-14
updated: 2026-04-14
source: experiment-looped-1Lx4-192D
confidence: high
verification: source
tags: [experiment, arithmetic, addition, looped-transformer, weight-sharing, parameter-efficiency]
related: [looped-transformers-parameter-efficiency, experiment-baseline-addition-reversed, depth-vs-width-arithmetic]
staleness_days: 0
---

## Insight

A looped transformer with 1 layer block looped 4 times (1Lx4), 4 heads, 192 dimensions achieves **99.2% exact accuracy** on 5-digit addition with only **453,696 parameters** — 87.3% fewer params than the 2L/384D baseline (3.56M params, 99.9%). Weight sharing via looping provides massive parameter efficiency with minimal accuracy loss.

## Evidence

Experimental results from `experiments/train.py` with `--architecture looped`:
- Config: 1L/4H/192D, n_loops=4, 50K train, 2K test, 80 epochs, batch_size=256
- Tokenizer: reversed, balanced carry sampling
- Final accuracy: 99.2% (vs baseline 99.9%)
- Final loss: 1.3563
- Total training time: 3451s (~58 min, with heavy CPU contention from parallel experiments)
- Result file: `add_5d_reversed_1L4H192D_looped_1Lx4_192D.json`

### Training Dynamics
- Slow start: 12% accuracy at epoch 10, 20% at epoch 20
- Rapid phase transition: 37% at epoch 25 → 62% at epoch 30 → 93% at epoch 40
- Near convergence: 97.1% at epoch 45, 98.3% at epoch 50, 99.2% at epoch 80
- Still improving at epoch 80 — more epochs might reach 99.5%+

### Per-Digit Accuracy (epoch 80)
| Digits | Accuracy |
|--------|----------|
| 1      | 100%     |
| 2      | 100%     |
| 3      | 99.2%    |
| 4      | 98.4%    |
| 5      | 99.3%    |
| 6 (carry) | 100%  |

### Key Features
- **Progressive loss**: Supervises output at each loop iteration, averaged across all 4 loops
- **Input injection**: Skip connection adds original embeddings after each loop
- **Weight sharing**: Single transformer block reused 4 times → 4x parameter efficiency

## Counter-argument (FUNGI)

This could be disproved if: (1) the accuracy plateau at 99.2% represents a hard ceiling from the looped architecture rather than insufficient training, or (2) the model memorizes patterns rather than learning the algorithm (would need OOD length generalization testing to confirm).

## Implications

Looped transformers are a viable approach for arithmetic tasks when parameter budget is constrained. The 1Lx4/192D config achieves near-baseline accuracy at ~13% of the parameter cost, suggesting weight sharing captures the iterative nature of carry propagation in addition.
