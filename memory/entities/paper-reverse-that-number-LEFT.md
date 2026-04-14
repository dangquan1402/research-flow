---
title: "Entity: Reverse That Number! / LEFT (Little-Endian Fine-Tuning)"
type: paper
created: 2026-04-13
tags: [arithmetic, reversed-order, little-endian, fine-tuning]
---

## Paper Details
- **Title:** Reverse That Number! Decoding Order Matters in Arithmetic Learning
- **ArXiv:** https://arxiv.org/abs/2403.05845
- **Year:** 2024

## Key Contributions
- Proposed LEFT (Little-Endian Fine-Tuning) for arithmetic in LLMs.
- Proved Big-Endian complexity is exponential (10^(2n+2)), Little-Endian is linear (n * 10^5).
- Achieved 11.1% overall accuracy improvement using only 5.2% of training tokens (addition/subtraction).
- Showed multiplication requires BOTH reversed order AND scratchpads.
- Error analysis: 2x reduction in intermediate product errors, 10x reduction in cumulative summation errors.

## Relevance to Our Project
Theoretical foundation for why reversed output is essential. Directly applicable to our training data format.
