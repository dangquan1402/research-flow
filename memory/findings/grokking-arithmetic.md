---
title: "Finding: Grokking Is Real but Fragile -- Weight Decay Is the Key Lever"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, grokking, regularization]
related: [optimizer-schedule-arithmetic]
staleness_days: 0
---

## Insight

Grokking -- sudden generalization long after overfitting -- is a genuine phenomenon in arithmetic transformers, mechanistically understood as the gradual formation of an algorithmic circuit (Fourier-based for modular arithmetic) that eventually displaces a memorization solution. Weight decay is the single most important hyperparameter for inducing grokking, as it makes memorization solutions energetically expensive, forcing the model toward compact generalizing circuits.

## Evidence

- Power et al. (2022) first documented grokking on modular arithmetic tasks with small transformers, showing test accuracy jumping from ~0% to ~100% thousands of steps after training accuracy reached 100%.
- Nanda et al. (2023) reverse-engineered the mechanism: the model learns a discrete Fourier transform circuit that projects inputs to rotations on a circle and composes them. Training proceeds through three phases: memorization, circuit formation, cleanup.
- Weight decay's role is mechanistically clear: memorization requires large weights to store lookup tables, while the generalizing Fourier circuit achieves the same accuracy with much smaller weights. Weight decay penalizes large weights, making memorization progressively more expensive.
- More weight decay = faster grokking (up to a point where training fails entirely).
- Weight decay more than halves the number of samples needed for generalization compared to other interventions.
- Grokking is contingent on hyperparameters: it disappears if model size, weight decay, data size, or other parameters are wrong.

## Reasoning

Grokking represents a competition between two solutions in weight space: (1) a memorization solution that stores input-output pairs explicitly (high weight norm) and (2) an algorithmic solution that implements the actual operation (low weight norm). Without regularization, the memorization solution is found first because it requires simpler gradient steps. Weight decay continuously penalizes the memorization solution, eventually making the algorithmic solution the lower-loss option. The "sudden" generalization is actually the final phase of a gradual transition, appearing sudden only because the last few memorization weights are pruned in a short window.

## Counter-arguments

- Grokking requires extremely long training (10-100x beyond overfitting), which may be impractical for production training.
- The phenomenon is best studied on modular arithmetic (finite groups), and its applicability to general integer arithmetic (unbounded) is less clear.
- Recent work (2025) questions whether grokking circuits transfer well to out-of-distribution inputs, suggesting the generalization may be more narrow than it appears.
- With good data format and curriculum, you may not need grokking at all -- the model can generalize during normal training.

## Implications

- **For our project**: Set AdamW weight decay to 0.1-1.0 (higher than typical NLP defaults of 0.01).
- Plan for long training runs if relying on grokking -- monitor validation loss well beyond training convergence.
- Use grokking as a diagnostic: if the model overfits and never grokks, weight decay may be too low or the model may be too large for the training set.
- Consider grokking dynamics when setting training budget: premature early stopping will miss generalization.
- For modular arithmetic subtasks, grokking is particularly reliable and well-understood.
