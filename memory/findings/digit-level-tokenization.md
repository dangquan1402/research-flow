---
title: "Finding: Digit-Level Tokenization Is a Prerequisite for Arithmetic Learning"
created: 2026-04-13
updated: 2026-04-13
source: tokenization-arithmetic-research
confidence: high
verification: source
tags: [tokenization, arithmetic, digit-level]
related: [bpe-hurts-arithmetic, reversed-digit-order]
staleness_days: 0
---

## Insight

Each digit of a number must be its own token for transformers to reliably learn arithmetic. This is the foundational tokenization requirement -- without it, no other technique (reversed output, scratchpads, position embeddings) can compensate. Digit-level tokenization enables the model to learn per-position operations as separable subtasks: digit-pair addition, carry detection, and carry propagation.

## Evidence

- **Lee et al. (ICLR 2024)** use character-level tokenization with NanoGPT (10.6M params) and achieve 100% accuracy on multi-digit addition with appropriate formatting. The paper explicitly states: "character-level tokenization and absolute position encoding."
- **Stolfo et al. (2024)** represent each digit as a separate token and achieve 99.999% accuracy on 5--15 digit addition/subtraction with only 2--3 layer models. Their mechanistic analysis shows attention heads learn to attend to specific digit positions in each operand.
- **Nogueira et al. (2021)** demonstrated that subword tokenization **fails** to learn 5-digit addition entirely, and even character-level struggles without additional position annotations.
- **McLeish et al. (2024)** and all subsequent work on Abacus Embeddings and Position Coupling assume digit-level tokenization as the baseline.

## Reasoning

Arithmetic algorithms operate digit-by-digit. When each digit is a separate token:
1. **Attention can align corresponding positions.** The model can learn that the i-th digit of operand A and the i-th digit of operand B combine to produce the i-th digit of the result (plus a carry).
2. **The vocabulary is minimal** (0--9 plus operators), so the embedding space is efficiently used for representing digit identity rather than memorizing multi-digit patterns.
3. **Carry propagation** becomes a token-to-token dependency that attention can represent, rather than an intra-token computation that must be done implicitly within an MLP.

Mechanistic interpretability work (Stolfo et al.) confirms that transformers decompose addition into three subtasks: (a) classifying each digit-pair sum as "definitely carry," "definitely no carry," or "uncertain" (TriCase); (b) resolving cascading carries (TriAdd); (c) computing final output digits. All three subtasks require access to individual digits as tokens.

## Counter-arguments

- **Tiny vocabulary may waste model capacity.** With only ~15 tokens, most of the embedding dimension is unused. Some researchers have proposed hybrid approaches with small multi-digit tokens.
- **Sequence length increases.** A 10-digit number requires 10 tokens instead of 3--4 BPE tokens. This increases compute quadratically in attention.
- **Real-world deployment friction.** Pretrained LLMs use BPE; switching to digit-level requires retraining or a specialized arithmetic head.

## Implications

For our project (small LLM for simple math):
- **Non-negotiable:** The tokenizer must separate every digit into its own token. This should be the first design decision.
- **Vocabulary design:** Tokens should be: `0 1 2 3 4 5 6 7 8 9 + - * = <pad> <eos>` (and possibly `<bos>`, space). Total ~18 tokens.
- **Sequence length budget:** For N-digit operands, input is ~2N+1 tokens (two operands + operator), output is ~N+1 tokens. Reasonable for integers up to ~20 digits.
- **Interaction with other techniques:** Digit-level tokenization is necessary but not sufficient. It must be combined with reversed output order and/or position-aware embeddings for best results.
