---
title: "Entity: Teaching Arithmetic to Small Transformers"
type: paper
created: 2026-04-13
tags: [arithmetic, tokenization, transformers, ICLR-2024]
---

## Paper Details
- **Title:** Teaching Arithmetic to Small Transformers
- **Authors:** Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, Dimitris Papailiopoulos
- **Venue:** ICLR 2024 (originally NeurIPS 2023 MATH-AI Workshop)
- **ArXiv:** https://arxiv.org/abs/2307.03381
- **Code:** https://github.com/lee-ny/teaching_arithmetic

## Key Contributions
- Demonstrated that character-level tokenization with reversed (LSB-first) output achieves 100% accuracy on multi-digit addition with ~2,500 samples using a 10.6M param NanoGPT.
- Showed plain format plateaus at ~85% while reversed and scratchpad formats reach 100%.
- Established sample complexity hierarchy: detailed scratchpad (1K) > simplified scratchpad (2K) > reversed (2.5K) >> plain (never).
- Connected training data requirements to low-rank matrix completion theory.

## Relevance to Our Project
Foundational reference for tokenization strategy decisions. Directly validates digit-level + reversed output as the baseline approach.
