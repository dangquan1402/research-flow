---
title: "Entity: Transformers Can Do Arithmetic with the Right Embeddings"
type: paper
created: 2026-04-13
tags: [arithmetic, abacus-embeddings, position-embeddings, length-generalization]
---

## Paper Details
- **Title:** Transformers Can Do Arithmetic with the Right Embeddings
- **Authors:** Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, Tom Goldstein
- **ArXiv:** https://arxiv.org/abs/2405.17399
- **Year:** 2024

## Key Contributions
- Introduced **Abacus Embeddings**: positional encodings that assign the same embedding to digits of identical significance.
- Achieved 99% accuracy on 100-digit addition, 6x length generalization (train 20, test 120).
- Combined with looped transformers + input injection for 99.1% OOD accuracy (87% error reduction).
- Extended to multiplication (15 digits) and array sorting (30 elements, 30-digit numbers).

## Relevance to Our Project
Primary reference for position-aware embedding design if we pursue length generalization.
