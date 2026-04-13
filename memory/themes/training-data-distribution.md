---
title: "Theme: Training Data Distribution Matters as Much as Volume"
created: 2026-04-13
updated: 2026-04-13
source: analysis
confidence: high
verification: analysis
tags: [training, sampling, data-distribution, augmentation]
synthesizes: [balanced-sampling-strategy, data-augmentation-arithmetic, curriculum-learning-arithmetic, scratchpad-chain-of-thought-arithmetic]
staleness_days: 0
---

## Pattern

For arithmetic transformers, **what examples you train on** matters as much as **how many** you train on. Uniform random sampling produces severely skewed data: 90% of numbers in [0, 999] have 3 digits, and long carry chains are exponentially rare. Yet carry propagation is exactly the skill the model needs most. Three data-side techniques address this without any architectural cost:

1. **Balanced carry sampling** — sample carry chain length uniformly first, then generate operands matching that length. Enables strong length generalization from scratch.
2. **Commutativity augmentation** — include both a+b and b+a. Effectively doubles the training set for free.
3. **Balanced digit sampling** — upweight shorter numbers to ensure coverage across all digit lengths.

Meanwhile, curriculum learning (progressive difficulty ordering) provides marginal benefit on top of good sampling and formatting. The autoregressive loss already creates implicit curriculum pressure.

## Supporting Findings

- **balanced-sampling-strategy**: Carry chain length follows a steep distribution under uniform sampling — long chains are exponentially rare but critical. Balanced carry sampling enabled the first strong length generalization on decimal addition from scratch. Non-negotiable for the data pipeline.
- **data-augmentation-arithmetic**: Commutativity augmentation (a+b → b+a), identity operations (a+0=a, a*1=a), and zero-padding provide meaningful accuracy improvements at negligible cost. MAMUT framework formalizes this.
- **curriculum-learning-arithmetic**: Helps sample efficiency but data format and sampling strategy matter more. The autoregressive loss provides implicit curriculum. Mixed curriculum (all difficulties simultaneously with weighted sampling) is better than strict sequential staging.
- **scratchpad-chain-of-thought-arithmetic**: For multiplication, the training data must include correct intermediate steps. The simplified scratchpad (carry + digit sum) offers the best accuracy-to-cost trade-off. Error propagation in intermediate steps is a risk.

## Contradictions

- **Balanced sampling vs. natural distribution**: Over-representing rare carry patterns distorts the natural distribution. If the model is later used in a context where most numbers are short and carries are rare, it may waste capacity on rare cases. **Resolution**: For a pure arithmetic model, accuracy on all cases matters more than matching natural distributions. Use balanced sampling.
- **Curriculum vs. simultaneous training**: Some practitioners say curriculum is essential; Lee et al. show it's secondary to format. **Resolution**: Use mixed curriculum (all difficulties, weighted toward target) rather than strict staging. The evidence favors simultaneous training with balanced sampling.

## Counter-arguments

- **Balanced carry sampling adds preprocessing cost**: Computing carry chains for each generated example is nontrivial. **Disproof condition**: If uniform random sampling with reversed output achieves >99% on the target digit range, the added complexity isn't justified.
- **Augmentation benefits diminish with scale**: Larger models may learn commutativity from data alone. **Disproof condition**: For our small model, augmentation is likely valuable.

## So What?

**For our training pipeline**:
1. Implement balanced carry sampling as the core data generator for addition/subtraction
2. Always include both a+b and b+a for commutative operations
3. Add 1-5% identity operations (a+0, a*1) for anchoring
4. Use balanced digit sampling (upweight shorter numbers)
5. For multiplication, generate scratchpad training data with a symbolic calculator
6. Log carry distribution and digit distribution as training diagnostics
7. Skip strict curriculum — use mixed difficulty with weighted sampling
