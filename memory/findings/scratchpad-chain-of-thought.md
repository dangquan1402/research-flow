---
title: "Finding: Scratchpads Are Essential for Multiplication but Optional for Addition"
created: 2026-04-13
updated: 2026-04-13
source: tokenization-arithmetic-research
confidence: high
verification: source
tags: [tokenization, arithmetic, scratchpad, chain-of-thought, multiplication]
related: [reversed-digit-order, digit-level-tokenization]
staleness_days: 0
---

## Insight

Scratchpad/chain-of-thought (CoT) tokenization -- where intermediate computation steps are included in the training target -- has an operation-dependent value. For addition and subtraction, reversed digit order alone is as effective or better than scratchpads. For multiplication, scratchpads are essential because the operation requires multi-step decomposition (partial products and accumulation) that cannot be compressed into a single pass. The key tradeoff is accuracy vs. sequence length: scratchpads increase generation length by 3--10x.

## Evidence

- **Lee et al. (ICLR 2024):** Detailed scratchpad achieves 100% accuracy with 1,000 samples vs. 2,500 for reversed-only and never for plain format (3-digit addition). But the detailed scratchpad increases sequence length substantially.
- **LEFT paper (2024):** "Little-Endian without step-by-step is smoother" in learning curves for addition/subtraction, suggesting reversed order captures the essential computational structure more efficiently than scratchpads. However, for multiplication, "both Little-Endian AND step-by-step proved essential" -- without step-by-step decomposition, multiplication learning failed entirely with 5K examples.
- **Nye et al. (2021) "Show Your Work":** Seminal paper demonstrating that scratchpads dramatically improve multi-step computation in language models, from long addition to program execution.
- **Merrill & Sabharwal (2023) "Expressive Power of Transformers with CoT":** Theoretically, intermediate generation tokens extend transformer computational power. Linear decoding steps (proportional to input length) add the ability to recognize all regular languages. Polynomial steps reach P-complete problems.
- **LEFT error analysis for multiplication:** Among 417 errors in prior SOTA, 140 occurred in intermediate products and 236 in cumulative summation. LEFT+scratchpad reduced these to 77 and 22 respectively (2x and 10x reductions).

## Reasoning

### Why addition does not need scratchpads (with reversed output)
Addition is fundamentally a single-pass algorithm: scan right-to-left, add digit pairs, propagate carry. With reversed output, this becomes a left-to-right single pass that fits naturally into autoregressive generation. Each output token depends on at most 5 input variables. No intermediate decomposition is needed.

### Why multiplication needs scratchpads
Multiplication of two N-digit numbers requires computing N partial products (each N+1 digits), then summing them with appropriate shifts. This is inherently a multi-step process:
1. Compute each partial product (N separate multiplications)
2. Align partial products (shift by position)
3. Sum all partial products (multi-operand addition with carries)

Without scratchpads, the model must perform all these steps implicitly within its forward pass. For a small transformer, this exceeds the computational capacity of the attention layers.

### Scratchpad design matters
Poorly designed scratchpads can actually hurt performance. Research shows that if the scratchpad format increases RASP-L program complexity (the measure of computational difficulty specific to transformers), it can hinder rather than help generalization. The scratchpad should decompose the problem into steps that are individually simple for the transformer.

## Counter-arguments

- **Sequence length cost is high.** A scratchpad for 5-digit multiplication might require 50--100 tokens of intermediate work. This impacts training compute (quadratic in attention) and inference latency.
- **Training data must include correct intermediate steps.** Generating scratchpad training data requires a symbolic calculator to produce step-by-step work, adding data pipeline complexity.
- **Diminishing returns with model scale.** Larger models may internalize multi-step computation without explicit scratchpads, though current evidence is that even large models benefit from CoT for arithmetic.
- **Alternative decompositions exist.** Instead of scratchpads in the output, one could use looped/recurrent transformer architectures that effectively give the model multiple passes, achieving similar computational depth without inflating sequence length.

## Implications

For our project:
- **Addition/subtraction:** Do NOT use scratchpads. Use reversed digit order instead -- it is simpler, more token-efficient, and equally (or more) effective.
- **Multiplication:** Use scratchpad format that decomposes into partial products and accumulation. Design the format to minimize unnecessary tokens while keeping each step simple.
- **Suggested multiplication scratchpad format:** Show each digit-by-digit partial product and running sum, with reversed digit order within each step.
- **Alternative for multiplication:** Consider a looped transformer architecture (weight sharing across layers) as a scratchpad alternative that does not inflate sequence length.
- **Training data generation:** Build a symbolic calculator that emits step-by-step multiplication work in the chosen scratchpad format.
