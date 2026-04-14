---
title: "Open Question: Does Grokking Apply to Unbounded Integer Arithmetic as Well as Modular Arithmetic?"
created: 2026-04-13
source: training-curriculum-research
priority: medium
tags: [arithmetic, grokking, modular-arithmetic, integer-arithmetic]
---

## Question

Nearly all grokking research uses modular arithmetic (finite groups with a fixed set of possible outputs). Our model operates on unbounded integer arithmetic where the output space grows with operand size. Does grokking still occur? If so, does the Fourier-based circuit mechanism still apply, or does the model learn a fundamentally different algorithm?

## Current State

- Grokking is well-documented on modular addition, subtraction, multiplication (mod p)
- Mechanistic understanding (Fourier circuits) is specific to modular/cyclic group structure
- Integer arithmetic has no cyclic structure -- carry propagation is the core algorithm, not rotation composition
- Some evidence of grokking-like behavior in general arithmetic training (delayed generalization after overfitting), but not systematically studied

## Why It Matters

If grokking doesn't reliably occur on integer arithmetic, our training strategy should not depend on it. We would need to rely more on data format, sampling, and curriculum instead of long training with high weight decay.

## Potential Approaches

- Run controlled experiments: train on integer addition with varying weight decay and training duration, monitor for grokking
- Compare training dynamics on modular vs. integer arithmetic with the same architecture
- Look for mechanistic signatures of grokking (weight norm decrease, circuit formation) in integer arithmetic models
