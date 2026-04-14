---
title: "Experimental Finding: Reversed Output Gives +2% Accuracy and 2x Learning Speed"
created: 2026-04-13
updated: 2026-04-13
source: experiment-tokenization-comparison
confidence: high
verification: source
tags: [experiment, arithmetic, tokenization, reversed-output, comparison]
related: [reversed-digit-order, digit-level-tokenization, experiment-baseline-addition-reversed]
staleness_days: 0
---

## Insight

In a controlled comparison on 5-digit addition, reversed (LSB-first) output achieves 99.9% accuracy vs 97.85% for plain (MSB-first) output, and reaches 95% accuracy in half the epochs (15 vs 30). The gap is largest for higher digit counts, consistent with carry propagation theory.

## Evidence

Controlled experiment using identical setup except output format:
- Model: 4L/4H/256D (3.16M params), 50K training samples, 50 epochs
- Reversed: 99.9% final (100% peak at epoch 45), 95% by epoch 15
- Plain: 97.85% final (never reaches 99%), 95% by epoch 30
- 6-digit accuracy: reversed 100% vs plain 98%
- Training time similar (~8-9 min each on MLX/M4)

## Reasoning

Reversed output aligns autoregressive left-to-right generation with right-to-left carry propagation, making each output token depend only on already-generated context (causal alignment). Plain format requires the model to implicitly predict carries before they're computed. The 2x learning speed advantage confirms the theoretical prediction of reduced sample complexity.

The plain format's 97.85% is stronger than some literature reports (~85%), possibly because balanced carry sampling provides enough carry-heavy examples for the model to learn carry prediction even in the anti-causal direction.

## Counter-arguments

- The 2% gap may narrow with more training epochs or data — plain might eventually approach reversed given enough compute
- With 50K training samples, the sample efficiency advantage may be understated — testing with smaller datasets would show a larger gap
- Single seed — should verify across seeds

## Implications

Reversed output is confirmed as the default choice for addition. The training efficiency benefit (2x faster to 95%) is as valuable as the accuracy ceiling improvement for rapid experimentation.
