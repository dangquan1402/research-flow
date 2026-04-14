---
title: "Paper: Teaching Arithmetic to Small Transformers"
created: 2026-04-13
updated: 2026-04-13
type: paper
tags: [arithmetic, transformer, nanoGPT, data-formatting]
staleness_days: 0
---

## Overview

- **Authors:** Nayoung Lee, Kartik Sreenivasan (and collaborators at Princeton)
- **Published:** 2023 (arXiv: 2307.03381), presented at MathAI workshop 2023
- **URL:** https://arxiv.org/abs/2307.03381

## Key Contribution

Demonstrated that data formatting (reversed output, chain-of-thought scratchpads, structured sampling) matters far more than model architecture for teaching arithmetic to small transformers. A NanoGPT model (6L/6H/384d, 10.6M params) achieves perfect addition when trained with reversed LSB-first output format.

## Architecture

- Decoder-only (NanoGPT), 6 layers, 6 heads, 384 embed dim, ~10.6M params
- Character-level tokenization, absolute positional encoding

## Referenced In

- [depth-vs-width-arithmetic]
- [minimum-viable-math-transformer]
- [data-formatting-over-architecture]
