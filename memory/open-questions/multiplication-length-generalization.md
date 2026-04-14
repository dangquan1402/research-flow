---
title: "Open Question: How to Achieve Strong Length Generalization for Multiplication"
created: 2026-04-13
source: training-curriculum-research
priority: high
tags: [arithmetic, multiplication, length-generalization]
---

## Question

Position coupling and balanced carry sampling solve length generalization for addition, but multiplication length generalization remains largely unsolved. Train set priming helps (5x3 -> 35x3) but requires seeing some long examples. What architectural or data-level technique can enable true zero-shot length generalization for multiplication?

## Current State

- Relative position embeddings fail for multiplication (Jelassi et al., 2023)
- Train set priming works but is a semi-supervised workaround
- Standard autoregressive loss does not encourage learning the right long-range dependencies for multiplication (as noted in recent work on "Why Can't Transformers Learn Multiplication?")
- Scratchpad/CoT training may help by decomposing multiplication into addition steps, but this is not well-validated for length generalization

## Why It Matters

Multiplication is one of our three target operations. If we can only do addition length generalization, the model's practical utility is limited.

## Potential Approaches to Investigate

- Decompose multiplication into repeated addition with scratchpad
- Use Universal Transformer (shared layers) for implicit recursion
- Explore "implicit chain-of-thought" approaches
- Consider separate multiplication-specific position coupling schemes
