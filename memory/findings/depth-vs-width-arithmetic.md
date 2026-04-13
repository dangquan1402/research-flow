---
title: "Finding: Width Dominates Depth for Small Arithmetic Transformers"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
confidence: high
verification: source
tags: [architecture, arithmetic, scaling, depth, width]
related: [minimum-viable-math-transformer, attention-heads-carry-operations]
staleness_days: 0
---

## Insight

For arithmetic tasks on small transformers (<100M params), wider architectures consistently outperform deeper ones. The optimal depth for arithmetic is surprisingly shallow (2-6 layers), and additional depth actively hurts performance due to gradient starvation. Width (embedding dimension and head count) should be prioritized over depth.

## Evidence

1. **"The Depth Delusion" (2026):** Beyond a critical depth D_crit ~ W^0.44 (sublinear in width), adding layers increases loss despite adding parameters. Optimal scaling: depth as D* ~ C^0.12, width as W* ~ C^0.34 — width should grow 2.8x faster than depth. Validated across 30 architectures from 17M to 7B params (R^2 = 0.922).

2. **Nikankin et al. (2024):** Achieved >99.999% accuracy on multi-digit addition with just 2 layers and 3 heads. The minimal depth works because addition is parallelizable across digit positions.

3. **Depth-Width Tradeoffs in Algorithmic Reasoning (2025):** For algorithmic tasks, with linear width, constant depth suffices. Wider models match deep model accuracy with faster train/inference time due to hardware parallelization.

4. **Position Coupling (Cho et al., 2024):** Theoretically proved a 1-layer transformer with coupled positions can solve addition for exponentially many digits. Depth is not the bottleneck — positional information is.

5. **Multiplication counterexample:** Even 12-layer models fail at multiplication because the problem is not depth but long-range dependency learning. Auxiliary losses (not more layers) solve this.

## Reasoning

Arithmetic operations like addition are inherently parallel across digit positions — each digit pair can be processed somewhat independently, with carry being the only sequential dependency. This parallelism maps naturally to width (more heads processing different positions simultaneously) rather than depth (more sequential processing steps). The carry chain is short (at most ~log(n) bits of state), so a few layers suffice to propagate carries.

The gradient starvation mechanism explains why depth hurts: in a D-layer transformer, gradient signal decays exponentially backward through layers during training. For small models where each layer has limited capacity, this decay means early layers receive vanishing gradients and fail to learn useful representations.

## Counter-arguments

- **Multiplication requires more depth:** Some papers show deeper models (4-12 layers) help with multiplication due to the need for iterative intermediate computations. However, this was better solved with auxiliary losses than raw depth.
- **Looped transformers blur the line:** Weight-tied looped transformers achieve effective depth without parameter depth, suggesting that computational depth (iterations) can matter even when parameter depth doesn't.
- **Task complexity matters:** For more complex mathematical reasoning beyond basic arithmetic, depth may become more important. These findings may not generalize to algebra or calculus.

## Implications

For our <100M parameter arithmetic LLM:
- **Target 2-4 layers** for addition/subtraction, up to 6 for multiplication
- **Invest parameter budget in width:** larger embedding dimensions (256-512) and more heads (4-8) rather than more layers
- **Consider looped transformers** if computational depth is needed: use 2-4 unique layers looped 4-8 times for parameter efficiency
- For a parameter budget of ~10M, a configuration like 2-4 layers / 256-384 embedding dim / 4-6 heads is well-supported by evidence
- Training will be faster on consumer GPU due to shallower architecture enabling better parallelization
