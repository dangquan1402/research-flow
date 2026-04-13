---
title: "LLM Architecture for Simple Math (Add/Sub/Mul)"
created: 2026-04-13
issue: GH-1
status: active
scope: "Architecture, tokenization, and training strategy for a small transformer that does addition, subtraction, multiplication"
success_criteria: "Concrete architecture spec with layer count, width, tokenization, training recipe — backed by literature"
---

## Research Question

What is the optimal architecture for a small LLM that reliably performs simple integer arithmetic (addition, subtraction, multiplication)?

We need concrete, implementable decisions on:
1. **Architecture** — How many layers? How wide? How many attention heads? What positional encoding?
2. **Tokenization** — Digit-level? Character-level? Reversed output? BPE? What helps the model learn carry/borrow?
3. **Training** — What data format? What curriculum? What optimizer/schedule? How to handle length generalization?

## Approach

Three parallel hypothesis investigations:
- **Hypothesis A** (GH-2): Tokenization strategies — the #1 lever for arithmetic accuracy
- **Hypothesis B** (GH-3): Architecture sizing — depth vs width tradeoffs
- **Hypothesis C** (GH-4): Training curriculum — data generation and training recipe

## Sources
- Teaching Arithmetic to Small Transformers (Lee et al.)
- nanoGPT / minGPT (Karpathy)
- Length Generalization in Arithmetic Transformers
- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- Any relevant papers on transformer arithmetic
