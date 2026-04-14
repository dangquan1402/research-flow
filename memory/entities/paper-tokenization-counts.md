---
title: "Entity: Tokenization Counts - Impact on Arithmetic in Frontier LLMs"
type: paper
created: 2026-04-13
tags: [tokenization, arithmetic, BPE, frontier-LLMs]
---

## Paper Details
- **Title:** Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs
- **Authors:** Aaditya K. Singh, DJ Strouse
- **ArXiv:** https://arxiv.org/abs/2402.14903
- **Year:** 2024

## Key Contributions
- Demonstrated that L2R 3-digit tokenization (GPT-3.5/4) drops accuracy to 8.25% on length-mismatch addition problems.
- Comma formatting trick (forcing R2L tokenization) raises accuracy from 68.5% to 97.8%.
- Catalogued tokenization approaches across model families (PaLM, LLaMA, GPT, Claude).
- Showed that model scale partially compensates for bad tokenization but does not eliminate it.

## Relevance to Our Project
Strongest evidence that BPE is harmful. Directly motivates our digit-level tokenizer design.
