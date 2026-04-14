---
title: "Paper: Progress Measures for Grokking via Mechanistic Interpretability"
type: paper
created: 2026-04-13
authors: [Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt]
year: 2023
venue: ICLR 2023
url: "https://arxiv.org/abs/2301.05217"
tags: [grokking, mechanistic-interpretability, modular-arithmetic, fourier-transform]
---

## Summary

Reverse-engineered the algorithm learned by a transformer trained on modular addition. Found the model implements a discrete Fourier transform circuit: projects inputs to rotations on a circle, composes rotations. Defined progress measures that split training into three phases: memorization, circuit formation, cleanup. Showed grokking is not sudden but the endpoint of gradual structural change in weights.

## Key Researcher: Neel Nanda

Leading researcher in mechanistic interpretability. His work on grokking provided the first clear mechanistic understanding of how transformers learn algorithmic tasks, connecting grokking to concrete circuit-level changes in the model.

## Relevance to Our Project

Provides actionable diagnostic: monitor weight norms and Fourier components of embeddings to track progress toward generalization during training. Also validates that weight decay is mechanistically the right lever for encouraging generalization.
