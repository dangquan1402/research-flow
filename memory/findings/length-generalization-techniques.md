---
title: "Finding: Position Coupling and Balanced Carry Sampling Are the Most Promising Length Generalization Techniques"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, length-generalization, positional-encoding]
related: [data-format-impact, curriculum-learning-arithmetic]
staleness_days: 0
---

## Insight

Length generalization (training on N-digit arithmetic, testing on N+k) remains the hardest challenge for arithmetic transformers. The two most effective techniques are: (1) position coupling, which assigns the same position IDs to digits of equal significance across operands and result, enabling 200-digit addition from 30-digit training; and (2) balanced carry sampling, which ensures training data covers all carry chain lengths uniformly, enabling the first strong length generalization on decimal addition from scratch.

## Evidence

- **Position coupling** (Cho et al., 2024): Assigns same position IDs to corresponding digit positions in operands and result. Theoretically proven: a 1-layer Transformer with coupled positions can solve addition with exponentially many digits. Empirically achieves >95% exact-match accuracy on 200-digit addition when trained on up to 30-digit addition.
- **Abacus Embeddings + FIRE** (McLeish et al., 2024): Learned positional embeddings encoding positions within numerical spans. Combined with FIRE encoding, enables strong arithmetic length generalization.
- **Train set priming** (Kazemnejad et al., 2023): Adding just 10-50 longer examples to training set enables generalization. Priming count scales logarithmically with training set size. Enables 5x3 digit multiplication to generalize to 35x3.
- **Balanced carry sampling**: Uniformly sampling carry chain length, then sampling operands with that carry length. First technique to achieve strong length generalization on decimal addition for transformers trained from scratch.
- **Attention Bias Calibration (ABC)**: A post-training calibration stage that learns proper attention biases, connected to RPE mechanisms. Near-perfect length generalization on some tasks.
- **Relative Position Embeddings**: Work for addition (5-digit to 15-digit) but fail for multiplication.
- **Universal Transformers** (shared layers): Recurrent structure helps length generalization, especially for modular arithmetic.

## Reasoning

Standard absolute positional embeddings encode "I am the 5th token" which is meaningless when the input length changes. For arithmetic, what matters is "I am the ones digit" or "I am the tens digit." Position coupling directly encodes this semantic meaning. Similarly, balanced carry sampling addresses the statistical gap: in random data, long carry chains are exponentially rare, so the model never learns to handle them. Both techniques address the root causes of length generalization failure rather than patching symptoms.

## Counter-arguments

- Position coupling requires task-specific knowledge to design the coupling scheme -- it doesn't generalize to arbitrary tasks.
- Priming with longer examples is a form of "cheating" -- the model sees examples of the target length during training, just very few of them.
- Most results are demonstrated on addition; multiplication length generalization remains largely unsolved.
- Abacus embeddings require knowing which tokens are digits, adding architectural complexity.

## Implications

- **For our project**: Implement position coupling as the primary positional encoding strategy. It's simple, theoretically grounded, and achieves the best results.
- Use balanced carry sampling regardless of other choices -- it's a data-side technique with no architectural cost.
- For multiplication, plan to use train set priming (add a small number of longer examples).
- Consider Universal Transformer architecture if length generalization is a hard requirement.
- Test length generalization explicitly during development: train on N digits, evaluate on 2N and 3N.
