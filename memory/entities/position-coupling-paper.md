---
title: "Paper: Position Coupling for Arithmetic Transformers"
created: 2026-04-13
updated: 2026-04-13
type: paper
tags: [arithmetic, transformer, positional-encoding, length-generalization]
staleness_days: 0
---

## Overview

- **Authors:** Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, Chulhee Yun
- **Published:** NeurIPS 2024 (arXiv: 2405.20671)
- **URL:** https://arxiv.org/abs/2405.20671
- **Code:** https://github.com/HanseulJo/position-coupling

## Key Contribution

Introduced Position Coupling — assigning same position IDs to digits of equal significance across operands and result. Achieved 6.67x length generalization (train on 1-30 digits, generalize to 200). Theoretical proof that 1-layer transformer + position coupling solves addition for exponentially many digits, while 1-layer without positional info cannot.

## Architecture

- 1-layer, 4-head (primary experiments)
- 6-layer, 8-head (larger experiments)
- 2-layer, 10-head for Nx2 multiplication (min embed dim >= 46)

## Referenced In

- [positional-encoding-arithmetic]
- [minimum-viable-math-transformer]
