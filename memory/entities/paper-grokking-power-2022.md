---
title: "Paper: Grokking - Generalization Beyond Overfitting on Small Algorithmic Datasets"
type: paper
created: 2026-04-13
authors: [Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra]
year: 2022
venue: ICLR 2022
url: "https://arxiv.org/abs/2201.02177"
tags: [grokking, generalization, modular-arithmetic, regularization]
---

## Summary

First paper to document and name the "grokking" phenomenon: models trained on small algorithmic datasets (particularly modular arithmetic) can suddenly generalize to the test set long after perfectly memorizing the training set. Showed that this occurs across many binary operations (modular addition, subtraction, multiplication, permutation composition) and that weight decay and dataset fraction are key controls.

## Relevance to Our Project

Establishes the foundational understanding of grokking. Key practical takeaway: set weight decay high enough and train long enough, and the model will transition from memorization to generalization. Critical for our training schedule planning.
