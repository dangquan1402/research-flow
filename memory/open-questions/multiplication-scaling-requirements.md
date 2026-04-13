---
title: "Open Question: How Does Multiplication Difficulty Scale with Digit Count?"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
priority: high
research_goal: "GH-3-architecture-sizing"
tags: [architecture, arithmetic, multiplication, scaling]
staleness_days: 0
---

## Question

Multiplication is known to be fundamentally harder than addition for transformers. Research shows 99.9% accuracy on 5-digit multiplication with a tiny transformer, but GPT-4 (billions of params) fails at 4x4 digit multiplication. What is the minimum architecture needed for N-digit multiplication, and how does it scale? Is there a digit-count ceiling for small transformers?

## Why It Matters

If our target is "multiplication of integers," we need to know what digit lengths are feasible. Can a <100M param model reliably multiply 5-digit numbers? 10-digit? The answer affects whether we need auxiliary losses, progressive training, or should limit the scope.

## Current Evidence

- 5-digit multiplication: 99.9% with tiny transformer + reversed format + progressive training
- 4x4 multiplication: 99% with 2 layers + auxiliary losses
- 12-layer models fail without auxiliary losses — depth alone doesn't help
- Long-range dependency problem: middle digits never receive correct gradients
- Position Coupling tested on Nx2 multiplication (not NxN)

## Possible Approaches

1. Systematic benchmark: test 2L/4H model on 2-digit through 10-digit multiplication
2. Evaluate auxiliary loss impact at each digit length
3. Test if looped transformers help (iterative refinement of partial products)
4. Consider limiting scope to "up to 5-digit multiplication" as realistic target
