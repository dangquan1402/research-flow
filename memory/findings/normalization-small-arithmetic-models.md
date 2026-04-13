---
title: "Finding: Pre-Norm Is Essential for Stable Small Transformer Training"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
confidence: high
verification: source
tags: [architecture, arithmetic, normalization, training-stability]
related: [minimum-viable-math-transformer]
staleness_days: 0
---

## Insight

Pre-norm (Pre-LN) is the clear choice for small arithmetic transformers. It eliminates learning rate warmup requirements, provides smooth training curves, and is especially advantageous in low-resource settings. Post-norm is unstable for small models and offers no compensating benefits at this scale.

## Evidence

1. **Training stability:** Pre-norm training curves are smooth and stable, while Post-norm shows loss spikes and unstable gradient norms. Pre-norm creates "an identity gradient highway that avoids the exponential decay seen in Post-LN architectures."

2. **Low-resource advantage:** "While POST-NORM performs better for high-resource NMT in the original base Transformer regime, PRE-NORM is both more stable and more competent in low-resource settings." Arithmetic models are definitionally low-resource (small data, small model).

3. **Warmup elimination:** Pre-norm allows warmup schedules to be "reduced or removed entirely." For rapid experimentation on consumer hardware, this simplifies hyperparameter tuning significantly.

4. **Performance:** On large-scale tasks, Pre-LN "outpaces Post-LN by 40% or more in wall-time to target loss, tolerates up to threefold higher learning rates, and maintains uniform, stable gradient profiles across all layers."

5. **Universal adoption:** GPT-2, GPT-3, LLaMA, Falcon, Mistral all use Pre-norm. NanoGPT (the base for arithmetic research) uses Pre-norm.

6. **Emerging hybrid:** HybridNorm (combining Pre- and Post-norm strengths) has been proposed but is not yet validated for small arithmetic models and adds implementation complexity.

## Reasoning

Small models have fewer layers and smaller hidden dimensions, making gradient flow more precarious. Post-norm applies layer normalization after the residual connection, which means the unnormalized residual path can accumulate large activations that destabilize training. Pre-norm normalizes before the sub-layer, ensuring the input to attention and FFN is always well-conditioned. The residual connection then adds a small, well-behaved perturbation.

For arithmetic specifically, training involves sharp phase transitions (grokking-like behavior where the model suddenly "gets" the algorithm). Pre-norm's stable gradients make these transitions more reliable and reproducible.

## Counter-arguments

- **Post-norm may yield slightly better final performance:** Some evidence suggests Post-norm achieves better converged performance in high-resource settings, as the lack of identity highway forces deeper feature transformation. However, this advantage disappears for small models.
- **RMSNorm vs LayerNorm:** Within pre-norm, RMSNorm (used by LLaMA) is simpler and faster than full LayerNorm, omitting the mean-centering step. For arithmetic models, the difference is negligible.

## Implications

For our arithmetic LLM:
1. **Use Pre-norm (Pre-LN)** — non-negotiable for training stability at this scale
2. **RMSNorm preferred** over LayerNorm for simplicity and speed (following LLaMA convention)
3. **No learning rate warmup needed** — simplifies training setup
4. **Higher learning rates tolerable** — enables faster convergence
5. **Standard implementation** — all reference codebases (nanoGPT, etc.) already use Pre-norm
