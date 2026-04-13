---
title: "Paper: The Depth Delusion"
created: 2026-04-13
updated: 2026-04-13
type: paper
tags: [architecture, scaling-laws, depth, width]
staleness_days: 0
---

## Overview

- **Published:** January 2026 (arXiv: 2601.20994)
- **URL:** https://arxiv.org/abs/2601.20994

## Key Contribution

Discovered that beyond a critical depth D_crit ~ W^0.44, adding layers increases loss despite adding parameters. Proposed architecture-conditioned scaling laws: optimal depth D* ~ C^0.12, optimal width W* ~ C^0.34. Width should grow 2.8x faster than depth. Validated across 30 architectures from 17M to 7B parameters (R^2 = 0.922). Core mechanism: gradient starvation through deep layers.

## Referenced In

- [depth-vs-width-arithmetic]
