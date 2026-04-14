---
title: "Entity: Position Coupling for Arithmetic Transformers"
type: paper
created: 2026-04-13
tags: [arithmetic, position-coupling, length-generalization, NeurIPS-2024]
---

## Paper Details
- **Title:** Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure
- **Authors:** Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, Chulhee Yun
- **Venue:** NeurIPS 2024
- **ArXiv:** https://arxiv.org/abs/2405.20671
- **Code:** https://github.com/HanseulJo/position-coupling

## Key Contributions
- Assigns identical position IDs to tokens of equal significance across operands and result.
- >95% exact-match accuracy on 200-digit additions trained on 1--30 digits (6.67x extrapolation).
- Theorem: 1-layer, 2-head transformer with coupled positions solves addition up to 2^floor((d-17)/2)-2 digits.
- Extends to N x 2 multiplication and 2D tasks (Minesweeper).
- Outperforms index hints in both efficiency and generalization range.

## Relevance to Our Project
State-of-the-art for length generalization. Consider if we need to handle numbers longer than training data.
