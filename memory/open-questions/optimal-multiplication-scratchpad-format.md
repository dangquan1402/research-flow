---
title: "Open Question: What Is the Optimal Scratchpad Format for Multiplication?"
created: 2026-04-13
tags: [multiplication, scratchpad, tokenization, open-question]
priority: high
---

## Question

What specific scratchpad format minimizes sequence length while maximizing multiplication accuracy for a small transformer? Should we use partial-product decomposition, digit-by-digit steps, or some hybrid?

## Context

The LEFT paper shows that multiplication requires both reversed output AND scratchpads -- reversed output alone fails. But the optimal scratchpad design is not settled. Lee et al. tested "simplified" vs. "detailed" formats for addition (where detailed was better), but the multiplication scratchpad design space is much larger.

Key design dimensions:
- How to decompose partial products (digit-by-digit vs. row-by-row)
- Whether to show running sums or only final accumulation
- Whether intermediate steps should also use reversed digit order
- How to handle alignment/shifting of partial products

## Why It Matters

Multiplication is the hardest of our three target operations. The scratchpad format directly determines both training data complexity and inference cost. A 5x5 digit multiplication could require 25--100 intermediate tokens depending on format.

## Potential Approaches to Investigate
1. Replicate LEFT paper's multiplication format exactly
2. Test RASP-L complexity of different decompositions
3. Ablate: partial products only vs. partial products + running sums
4. Compare with looped transformer (no scratchpad) as alternative
