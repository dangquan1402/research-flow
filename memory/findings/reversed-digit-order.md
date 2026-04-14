---
title: "Finding: Reversed (LSB-First) Output Order Dramatically Improves Arithmetic Accuracy"
created: 2026-04-13
updated: 2026-04-13
source: tokenization-arithmetic-research
confidence: high
verification: source
tags: [tokenization, arithmetic, reversed-order, carry-propagation]
related: [digit-level-tokenization, bpe-hurts-arithmetic]
staleness_days: 0
---

## Insight

Outputting the answer least-significant digit first (reversed/little-endian order) is one of the highest-impact formatting changes for arithmetic transformers. It aligns the autoregressive left-to-right generation order with the right-to-left carry propagation order of standard addition, transforming carry dependencies from anti-causal to causal. This single change reduces sample complexity by 4--10x and raises accuracy ceilings from ~85% to 100%.

## Evidence

- **Lee et al. (ICLR 2024):** Plain (MSB-first) format plateaus at ~85% accuracy even with 10,000 training samples. Reversed format reaches 100% accuracy with ~2,500 samples -- a 4x improvement in sample efficiency. Under noisy conditions, reversed maintains 81.26% exact accuracy vs. 49.88% for plain.
- **LEFT paper (arXiv:2403.05845):** Little-Endian Fine-Tuning achieves 11.1% overall accuracy improvement over prior SOTA while using only 5.2% of training tokens for addition/subtraction. Joint training results: 94.4% overall (LEFT) vs. 83.3% (prior SOTA). Multiplication specifically jumps from 52.8% to 88.5%.
- **RevOrder (Shen, 2025):** Achieves 100% accuracy on Big-bench arithmetic for addition, subtraction, and multiplication with reversed output.
- **Theoretical proof (LEFT paper):** Big-Endian learning complexity is C_Big >= 10^(2n+2) (exponential in digit count n). Little-Endian complexity is C_Little <= n * 10^5 (linear in n). This is because each output digit in Little-Endian depends on only 5 variables (a_i, a_{i-1}, b_i, b_{i-1}, c_{i-1}) vs. all preceding digits in Big-Endian.

## Reasoning

The fundamental issue is a mismatch between generation order and computation order:

1. **Standard addition algorithm:** Humans add right-to-left. The carry from position i must be known before computing position i+1.
2. **Autoregressive generation:** Transformers generate left-to-right. Token at position t can only attend to tokens at positions < t (causal masking).
3. **The conflict:** In MSB-first output, the model must predict the most-significant digit first, but this digit depends on carries that propagate from the least-significant digit. The model must somehow "look ahead" through all digit positions.
4. **The resolution:** Reversing output order means the first generated digit is the least-significant one (no carry dependency). Each subsequent digit depends only on the previous carry, which has already been computed and is in context.

This is formalized by the RASP-L program complexity measure: the reverse-order addition algorithm has shorter RASP-L program length than the forward-order one, meaning it is more naturally expressible within the transformer's computational model.

Attention visualization (LEFT paper, Figure 4) confirms that layer 22 of a fine-tuned LLM shows patterns consistent with re-computing carries from previous digits -- exactly the expected behavior of right-to-left carry propagation in a left-to-right generation framework.

## Counter-arguments

- **Reversed output is unnatural for humans.** At inference time, the output must be re-reversed for readability. This is trivial computationally but adds a post-processing step.
- **Not needed if scratchpads are used.** The scratchpad format explicitly provides carry information, potentially making reversal redundant. However, LEFT shows that reversed-without-scratchpad outperforms scratchpad-without-reversal for addition/subtraction, and reversal is simpler.
- **Multiplication still needs scratchpads.** Reversed output alone is insufficient for multiplication -- LEFT multiplication without step-by-step decomposition fails entirely with limited training data.
- **Some position-aware methods may render it unnecessary.** Position Coupling (Cho et al. 2024) achieves strong length generalization with standard output order, though it modifies the architecture rather than the data format.

## Implications

For our project:
- **Strongly recommended** for addition and subtraction: reverse the output digit order during training and inference.
- **For multiplication:** Use reversed output order combined with scratchpad/chain-of-thought intermediate steps.
- **Implementation:** During data generation, reverse the digits of the answer string. At inference, reverse the model's output. E.g., training pair: `1 2 3 + 4 5 6 = 9 7 5` (answer 579 reversed).
- **Training efficiency:** Expect 4--10x reduction in required training data compared to standard order, which is significant for our compute budget.
- **Interaction with position embeddings:** Reversed order and position-aware embeddings (Abacus/Position Coupling) are complementary. Reversed order handles the carry direction problem; position embeddings handle length generalization.
