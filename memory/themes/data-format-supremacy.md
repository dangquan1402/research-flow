---
title: "Theme: Data Format Supremacy — Formatting Decisions Outweigh Architecture Choices"
created: 2026-04-13
updated: 2026-04-13
source: analysis
confidence: high
verification: analysis
tags: [tokenization, data-format, architecture, training]
synthesizes: [digit-level-tokenization, reversed-digit-order, bpe-hurts-arithmetic, data-formatting-over-architecture, data-format-impact, scratchpad-chain-of-thought]
staleness_days: 0
---

## Pattern

Across all three hypothesis branches (tokenization, architecture, training), one finding recurs with overwhelming consistency: **how you format the data matters more than how you build the model.** Reversing output to LSB-first triggers a phase transition from ~85% to 100% accuracy on the same architecture. Digit-level tokenization is the prerequisite without which nothing else works. BPE tokenization can crater accuracy to 8.25%. Meanwhile, scaling from 2 layers to 12 layers, or from 3M to 100M parameters, produces marginal gains by comparison.

The hierarchy of impact, from highest to lowest:
1. **Digit-level tokenization** (non-negotiable prerequisite — binary gate)
2. **Reversed (LSB-first) output** (85% → 100% accuracy, 4-10x sample efficiency)
3. **Scratchpad for multiplication** (essential for multi-step operations)
4. **Balanced carry sampling** (data distribution, not format, but same principle)
5. **Architecture changes** (incremental improvements only after the above)

## Supporting Findings

- **digit-level-tokenization**: Each digit must be its own token. Without this, attention cannot align place values, and no other technique compensates. This is the foundational gate.
- **reversed-digit-order**: LSB-first output aligns autoregressive generation with carry propagation direction. Reduces learning complexity from exponential (10^(2n+2)) to linear (n * 10^5) in digit count. The single highest-ROI change.
- **bpe-hurts-arithmetic**: BPE creates inconsistent digit boundaries and misaligned place values. The "comma trick" (forcing R2L tokenization) raises GPT-3.5 from 68.5% to 97.8%, proving tokenization is the bottleneck.
- **data-formatting-over-architecture**: Same NanoGPT architecture goes from 85% (plain) to 100% (reversed) with only a data format change. No architecture modification produces this magnitude of improvement.
- **data-format-impact**: Plain format *never* reaches 100% accuracy on multi-digit addition regardless of training budget. This is a fundamental constraint, not a sample efficiency issue.
- **scratchpad-chain-of-thought**: For multiplication, scratchpad externalizes multi-step computation that exceeds single-pass transformer capacity. Without it, multiplication fails entirely with limited data.

## Contradictions

- **Scratchpad vs. reversed**: Two findings address the same problem (carry direction) with different solutions. For addition, reversed output wins — it's simpler and equally effective. For multiplication, scratchpad is necessary because the problem requires multi-step decomposition that reversal alone cannot provide. **Resolution**: These are complementary, not competing. Use reversed output for add/sub, reversed output + scratchpad for multiplication.
- **Position-aware embeddings partially obviate reversed output**: Position Coupling achieves strong results with standard (MSB-first) output order. However, reversed output is a data-side change with zero architectural cost, while Position Coupling requires custom position ID assignment. **Resolution**: Use both — reversed output handles carry direction, position encoding handles length generalization.

## Counter-arguments

- **Scale may eventually overcome format**: GPT-4 shows reduced (but persistent) tokenization effects versus GPT-3.5. At sufficient scale, models may learn to internally reverse or decompose. **Disproof condition**: If a >100B parameter model with BPE achieves >99% on arbitrary-length addition without special formatting, this theme weakens. Current evidence says this hasn't happened.
- **Format-dependence limits generality**: A model trained on reversed output cannot natively handle MSB-first queries. This creates deployment friction. **Disproof condition**: If a single format achieves >99% on all operations without post-processing, format supremacy becomes less actionable.

## So What?

**For our architecture spec**: Lock in data format decisions before any architecture decisions. Specifically:
1. Digit-level tokenization with vocabulary {0-9, +, -, *, =, PAD, EOS} (~16 tokens)
2. LSB-first (reversed) output for all operations
3. Scratchpad intermediate steps for multiplication only
4. Only then choose layer count, head count, embedding dimension — these are secondary levers
