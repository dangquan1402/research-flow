---
title: "Paper: Transformers Can Do Arithmetic with the Right Embeddings"
created: 2026-04-13
updated: 2026-04-13
type: paper
tags: [arithmetic, transformer, abacus-embeddings, looped-transformer, length-generalization]
staleness_days: 0
---

## Overview

- **Authors:** Sean McLeish et al.
- **Published:** NeurIPS 2024 (arXiv: 2405.17399)
- **URL:** https://arxiv.org/abs/2405.17399

## Key Contribution

Introduced Abacus Embeddings — learned embeddings that encode digit position relative to number start with randomized offsets during training. Combined with input injection and looped transformers, achieved 6x length generalization (train on 20 digits, generalize to 100+). Showed looped transformers (weight-tied layers) are highly effective for arithmetic.

## Architecture

- Standard transformer: 16 decoder layers
- Looped variants: 1x16 through 16x1 configurations
- Parameters: 12M (looped) to 122M (standard)

## Referenced In

- [positional-encoding-arithmetic]
- [looped-transformers-parameter-efficiency]
- [minimum-viable-math-transformer]
