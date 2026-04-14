---
title: "Experimental Finding: 4L/4H/256D Transformer Achieves 99.9% on 5-Digit Addition"
created: 2026-04-13
updated: 2026-04-13
source: experiment-baseline-addition
confidence: high
verification: source
tags: [experiment, arithmetic, addition, reversed-output, baseline]
related: [minimum-viable-math-transformer, reversed-digit-order, digit-level-tokenization, balanced-sampling-strategy]
staleness_days: 0
---

## Insight

A 4-layer, 4-head, 256-dim transformer (3.16M params) achieves 99.9% exact accuracy on 5-digit addition with reversed (LSB-first) output format and balanced carry sampling, trained in under 9 minutes on Apple M4 with MLX. Peak accuracy of 100% was reached at epoch 45 of 50.

## Evidence

Experimental results from `experiments/train_math_transformer.py`:
- Config: 4L/4H/256D, 50K training samples, 2K test samples, 50 epochs
- Final accuracy: 99.9% (1998/2000 correct)
- Peak accuracy: 100% at epoch 45
- All digit counts (1-6) at 100% by epoch 50
- Training time: 517s on MLX with Apple M4 Metal GPU
- Loss converged smoothly from 2.03 to 1.30

## Reasoning

This validates several research findings simultaneously:
1. **Reversed output** enables near-perfect addition (predicted by Lee et al., LEFT paper)
2. **Small models suffice** — 3.16M params vs. the 10M parameter budget (predicted by minimum-viable finding)
3. **Balanced carry sampling** ensures uniform accuracy across digit counts
4. **Pre-norm RMSNorm** provides stable training (no loss spikes observed)
5. **MLX on M4** is a viable training platform for arithmetic transformers

## Counter-arguments

- Test set is generated from the same distribution as training — does not test length generalization
- 2000 test samples may miss rare failure modes
- Single seed (42) — results should be verified across seeds

## Implications

This baseline confirms the research-recommended configuration works. Next experiments should compare against plain (non-reversed) tokenization and test depth/width tradeoffs.
