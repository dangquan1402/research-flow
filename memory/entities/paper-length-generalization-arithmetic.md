---
title: "Paper: Length Generalization in Arithmetic Transformers"
type: paper
created: 2026-04-13
authors: [Samy Jelassi, Stéphane d'Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, François Charton]
year: 2023
venue: NeurIPS 2023
url: "https://arxiv.org/abs/2306.15400"
tags: [length-generalization, arithmetic, positional-encoding, train-set-priming]
---

## Summary

Systematic study of length generalization in arithmetic transformers. Found that relative position embeddings enable addition generalization (5-digit to 15-digit) but fail for multiplication. Proposed train set priming: adding 10-50 long examples enables multiplication generalization (5x3 to 35x3). Priming count scales logarithmically with training set size.

## Relevance to Our Project

Establishes the baseline for what works and what doesn't in length generalization. Train set priming is a practical, low-cost technique we should implement for multiplication.
