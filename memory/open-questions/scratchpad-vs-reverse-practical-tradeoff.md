---
title: "Open Question: When Does Scratchpad Justify Its Sequence Length Cost Over Reverse Format?"
created: 2026-04-13
source: training-curriculum-research
priority: medium
tags: [arithmetic, data-format, scratchpad, inference-cost]
---

## Question

Reverse (Little-Endian) format achieves near-perfect accuracy on basic arithmetic with minimal sequence overhead. Scratchpad achieves even higher accuracy and better sample efficiency but at 3-10x sequence length cost. For our specific use case (addition, subtraction, multiplication of integers), at what point does scratchpad become necessary? Is reverse format sufficient for all three operations?

## Current State

- RevOrder achieves 100% on addition, subtraction, multiplication (low digit count) with reverse format alone
- Scratchpad is clearly superior for very long digit counts and complex operations
- No direct comparison exists for the specific crossover point where reverse format begins to fail
- For multiplication specifically, reverse format alone may not be sufficient for larger digit counts

## Why It Matters

This directly impacts our training compute budget and inference latency. If reverse format is sufficient for our target digit range (e.g., up to 8 digits), we save 3-10x on both training and inference.

## Potential Approaches

- Empirical ablation: train with both formats on our target operations and digit ranges
- Start with reverse format, measure where accuracy drops, switch to scratchpad for those cases
- Consider a hybrid: reverse for addition/subtraction, scratchpad for multiplication
