---
title: "Open Question: Position Coupling vs Abacus Embeddings — Which Is Better?"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
priority: medium
research_goal: "GH-3-architecture-sizing"
tags: [positional-encoding, length-generalization, arithmetic]
staleness_days: 0
---

## Question

Position Coupling (6.67x generalization, NeurIPS 2024) and Abacus Embeddings (6x generalization, NeurIPS 2024) both achieve excellent length generalization for arithmetic, but they were evaluated in different papers with different base architectures and training setups. Which performs better in a controlled head-to-head comparison? Can they be combined?

## Why It Matters

We need to choose a positional encoding strategy for our model. Both are strong candidates with similar claimed performance. The implementation differences are significant: Position Coupling modifies position ID assignment (zero-parameter), while Abacus Embeddings add learned parameters.

## Current Evidence

- Position Coupling: 1-30 digits -> 200 digits (6.67x), tested on 1-6 layer models
- Abacus Embeddings: 20 digits -> 100+ digits (6x), tested with looped transformers
- McLeish et al. note Abacus Embeddings "can be combined with FIRE and RoPE"
- No published head-to-head comparison exists

## Possible Approaches

1. Implement both on the same base architecture and benchmark
2. Try combining them: Position Coupling for position IDs + Abacus for additional digit-significance signal
3. Test with and without looped transformers to isolate the PE contribution
