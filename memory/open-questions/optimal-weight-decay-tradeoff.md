---
title: "Open Question: What Is the Optimal Weight Decay for Balancing Grokking Speed and Final Performance?"
created: 2026-04-13
source: training-curriculum-research
priority: medium
tags: [arithmetic, grokking, optimizer, weight-decay]
---

## Question

Higher weight decay accelerates grokking (transition from memorization to generalization) but too much weight decay prevents learning entirely. What is the optimal weight decay value for our specific model size (~10M params) and training set size? Is the optimal value for grokking the same as the optimal value for final test accuracy?

## Current State

- Literature uses weight decay values ranging from 0.01 to 1.0
- Power et al. showed weight decay is the most impactful hyperparameter but didn't identify a universal optimal value
- The optimal likely depends on model size, data size, and training duration in complex ways
- No systematic study exists for the specific setting of small transformers on integer (not modular) arithmetic

## Why It Matters

Getting weight decay wrong could mean either: (1) the model memorizes and never generalizes, or (2) the model is too regularized to learn the task at all. This is likely the most important hyperparameter to tune.

## Potential Approaches

- Systematic sweep over weight_decay in [0.01, 0.05, 0.1, 0.5, 1.0] with our specific architecture
- Monitor weight norms during training as a diagnostic
- Consider adaptive weight decay schedules (start low, increase over training)
