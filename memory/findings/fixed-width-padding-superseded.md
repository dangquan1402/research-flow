---
title: "Finding: Fixed-Width Zero-Padding Is Useful but Largely Superseded"
created: 2026-04-13
updated: 2026-04-13
source: tokenization-arithmetic-research
confidence: medium
verification: source
tags: [tokenization, arithmetic, zero-padding, formatting]
related: [digit-level-tokenization, index-hints-position-markers]
staleness_days: 0
---

## Insight

Zero-padding numbers to a fixed width (e.g., `0123 + 0456 = 0579`) was an early technique to ensure strict digit alignment across operands and results. While effective, it has been largely superseded by position-aware embeddings (Abacus, Position Coupling) that achieve the same alignment without wasting tokens on leading zeros. Zero-padding remains a simple, no-architecture-change baseline worth considering for projects that cannot modify the embedding layer.

## Evidence

- **Lee et al. (ICLR 2024):** Initially used zero-padding for operands and outputs but later found that wrapping samples with `$` delimiters was equally effective with less overhead.
- **ABA-fixed format (2024):** Guarantees strict alignment of corresponding digits across operands and sum. ABA-var preserves weaker alignment but works with standard inference (no padding at test time).
- **McLeish et al. (2024):** Abacus Embeddings "do not require padding each operand to the same length with zeros" -- explicitly positioned as an improvement over zero-padding.
- **Several papers note:** A single padding symbol can effectively replace zero-padding, suggesting the alignment signal, not the zeros themselves, is what matters.

## Reasoning

Zero-padding works by ensuring every number has the same number of digit tokens. This means:
1. The ones digit of every number is always at the same sequence position.
2. The tens digit is always at the same position, etc.
3. Attention can learn position-specific operations that transfer across all numbers.

The limitation is that it requires knowing the maximum number width at training time and wastes tokens on leading zeros. For 20-digit capacity, a 3-digit number like 123 would be represented as `00000000000000000123` -- 17 wasted tokens per operand.

Position-aware embeddings achieve the same alignment by encoding digit significance directly in the embedding or position IDs, without any padding tokens. They are strictly superior in both token efficiency and generalization ability.

## Counter-arguments

- **Simplicity.** Zero-padding requires no architectural changes -- just data formatting. It works with any standard transformer.
- **Predictable sequence length.** Fixed-width padding makes batch processing trivial since all sequences have the same length.
- **No implementation risk.** Position-aware embeddings require custom code that could have bugs. Zero-padding is trivially correct.

## Implications

For our project:
- **Consider as a baseline.** Zero-padding is the simplest way to ensure digit alignment. Use it for initial experiments before implementing Abacus Embeddings or Position Coupling.
- **Choose a reasonable max width.** For "simple math with integers," padding to 10--12 digits covers most practical use cases without excessive waste.
- **Upgrade path:** Start with zero-padding + reversed output, then swap in Abacus Embeddings if length generalization is needed.
