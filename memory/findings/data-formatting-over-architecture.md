---
title: "Finding: Data Formatting Matters More Than Architecture for Arithmetic"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
confidence: high
verification: source
tags: [architecture, arithmetic, data-formatting, training]
related: [minimum-viable-math-transformer, positional-encoding-arithmetic]
staleness_days: 0
---

## Insight

For arithmetic transformers, data formatting choices (reversed output, chain-of-thought scratchpads, structured sampling) yield larger accuracy improvements than architectural changes. Reversing output to LSB-first triggers a phase transition from ~85% to ~100% accuracy with the same model. This is the single most impactful design decision for arithmetic learning.

## Evidence

1. **Reverse formatting (Lee 2023):** Plain MSB-first format plateaus at ~85% accuracy with 10K training samples on NanoGPT (6L/6H/384d). Reversing output to LSB-first achieves "a distinct phase transition around 2500 train samples where it learns addition perfectly." Same architecture, same hyperparameters — only the data format changed.

2. **Chain-of-thought scratchpad (Lee 2023):** Detailed scratchpad formatting (showing intermediate digit sums and carries) reduced samples needed to 1,000 for perfect 3-digit addition, versus 2,000+ for reverse-only. Simplified scratchpad was nearly as effective.

3. **Structured sampling (Lee 2023):** Balancing training data by digit frequency and carry operations significantly outperformed random sampling. This ensures the model sees sufficient examples of carry cascades, which are underrepresented in random samples.

4. **Reversed format for multiplication (2024):** "Reversing the answer digits to enable the transformer can better calculate the carry" — same principle applies to multiplication, and was one of three critical factors (alongside progressive training and depth) for achieving 99.9% on 5-digit multiplication.

5. **Phase transition theory:** Addition tables decompose as rank-2 matrices, explaining sharp accuracy transitions at ~O(n) samples. This low-rank structure is what makes reverse formatting work — it enables the model to learn local functions rather than requiring global algorithms.

## Reasoning

MSB-first output forces the model to predict the most significant digit first, but that digit depends on carry information from ALL less significant positions. This requires the model to "look ahead" through the entire number — a global dependency. LSB-first output means the model predicts the least significant digit first, which depends only on the two input digits at that position. Each subsequent digit depends only on the previous carry — a local dependency chain. This is fundamentally easier for autoregressive models.

Chain-of-thought formatting makes the carry state explicit in the training data, so the model doesn't need to learn to represent carries internally — they're in the visible sequence. This is essentially giving the model external working memory.

## Counter-arguments

- **Reverse format is unnatural:** For deployment, users expect MSB-first output. The model would need a post-processing step to reverse the output, or be trained with a non-autoregressive decoder.
- **Scratchpad increases sequence length:** Chain-of-thought formatting makes sequences 3-5x longer, increasing training time and memory. For long numbers, this may exceed context windows.
- **Not all tasks benefit equally:** Reverse formatting helps addition more than multiplication. For multiplication, progressive training and auxiliary losses are relatively more important.
- **Generalization unclear:** These formatting tricks work great on clean arithmetic but may not help with embedded arithmetic in natural language contexts.

## Implications

For our arithmetic LLM:
1. **Use reversed output format** as the default — it's free and yields the largest single improvement
2. **Consider chain-of-thought** for the training set if we need faster convergence or fewer training samples
3. **Implement structured sampling** to ensure carry/borrow cases are well-represented in training data
4. **Plan for output reversal** in the inference pipeline if user-facing output needs to be MSB-first
5. **Training data budget:** With reversed format, ~2500 samples per digit length may suffice for perfect addition. Without it, 10K+ samples still won't reach 100%.
