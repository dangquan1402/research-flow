---
title: "Mixed-Operation Training: Single Model Handles Add+Sub+Mul"
created: 2026-04-14
updated: 2026-04-14
confidence: high
verification: source
tags: [experiment, arithmetic, mixed-ops]
---

## Finding

A single 2-layer transformer (2L/4H/384D, 2.78M params) trained on mixed arithmetic operations achieves near-perfect accuracy on all three operations simultaneously:

- **Addition (5-digit): 99.10%**
- **Subtraction (5-digit): 99.70%**
- **Multiplication (3-digit w/ scratchpad): 97.45%**

No catastrophic forgetting or significant interference between operations. The model learns to dispatch based on the operator token (+, -, *) and apply the correct output format (reversed for add/sub, scratchpad+reversed for mul).

## Evidence

- Experiment: `experiments/results/mixed_5d_reversed_2L4H384D_mixed_2L384D.json`
- 50,000 training examples (equal proportions), 80 epochs, MLX framework
- Peak overall accuracy: 99.50% at epoch 60; peak mul: 99% at epochs 50 and 75
- Final accuracy slightly below peak due to continued training past optimum

## Key Observations

1. **Learning order**: Sub > Add > Mul. Subtraction converges first (93% by epoch 10), addition next (100% by epoch 15), multiplication last (89% by epoch 30)
2. **No catastrophic interference**: All operations maintain high accuracy throughout training
3. **Mul format matters**: Scratchpad automatically enabled for mul examples; add/sub use plain reversed format. The model correctly identifies which format to use based on the operator token
4. **Digit cap important**: Mul capped at 3 digits (vs 5 for add/sub) since multiplication generates much longer output sequences
5. **Optimal stopping**: Best checkpoint around epoch 45-50; later epochs show slight mul degradation

## What Would Disprove This?

- If mul accuracy remains below 98% even with 4L/256D architecture (would suggest mixed training inherently limits mul performance)
- If separate single-op models significantly outperform the mixed model (would suggest interference costs)
- Testing with >3-digit multiplication in mixed mode — if performance degrades sharply, the model may be memorizing rather than generalizing

## Related

- [experiment-scratchpad-2L384D](experiment-scratchpad-2L384D.md) — scratchpad mul baseline
- [experiment-subtraction-5digit](experiment-subtraction-5digit.md) — subtraction baseline
- [experiment-baseline-addition-reversed](experiment-baseline-addition-reversed.md) — addition baseline
