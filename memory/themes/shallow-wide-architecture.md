---
title: "Theme: Shallow and Wide — 2-4 Layers With Many Heads Is the Optimal Arithmetic Architecture"
created: 2026-04-13
updated: 2026-04-13
source: analysis
confidence: high
verification: analysis
tags: [architecture, scaling, depth, width, attention-heads]
synthesizes: [depth-vs-width-arithmetic, minimum-viable-math-transformer, attention-heads-carry-operations, looped-transformers-parameter-efficiency, normalization-small-arithmetic-models]
staleness_days: 0
---

## Pattern

Arithmetic in transformers is fundamentally a **parallel, position-wise operation** with short sequential dependencies (carry chains). This maps to **width** (many heads processing digit positions simultaneously) rather than **depth** (many sequential processing steps). The evidence converges on 2-4 layers as optimal, with additional depth actively hurting due to gradient starvation. The carry mechanism requires exactly 3 specialized heads, and 4-6 heads suffice for mixed operations. Looped transformers offer an escape hatch: 2 unique layers looped 4-8 times provide effective depth without parameter depth.

## Supporting Findings

- **depth-vs-width-arithmetic**: "The Depth Delusion" (2026) shows width should grow 2.8x faster than depth. Beyond D_crit ~ W^0.44, more layers increase loss. Optimal for arithmetic: 2-4 layers with 256-384 embedding dimension.
- **minimum-viable-math-transformer**: 2-layer/3-head models achieve >99.999% on addition (Nikankin 2024). 10.6M-param NanoGPT (6L/6H/384d) is overprovisioned — the key was data formatting, not size. Conservative spec: 4L/4H/384d (~10M). Aggressive: 2L/4H/256d (~2-3M).
- **attention-heads-carry-operations**: Mechanistic interpretability reveals exactly 3 head roles: base digit addition (SA), carry detection (ST), carry cascade (SV). This 3-head pattern is universal across independently trained models. 4-6 heads for mixed operations.
- **looped-transformers-parameter-efficiency**: 2 layers x 8 loops = 16 effective layers at 2-layer parameter cost. Combined with Abacus Embeddings: 92.9% → 99.1% OOD accuracy. Input injection and progressive loss are critical for this to work.
- **normalization-small-arithmetic-models**: Pre-norm (RMSNorm) is essential for stable training at this scale. Eliminates warmup requirement and tolerates higher learning rates.

## Contradictions

- **Multiplication may need more depth**: Some papers show 4-12 layers help multiplication. However, the depth-vs-width finding and the auxiliary losses finding both indicate that multiplication's difficulty comes from multi-step computation, not raw depth. **Resolution**: Looped transformers (2 unique layers, 4-8 loops) provide computational depth for multiplication without parameter depth. Alternatively, scratchpad format externalizes the multi-step computation.
- **1-layer sufficiency vs. 2-4 layer recommendation**: Position Coupling proves a 1-layer transformer can solve addition for exponentially many digits. But this is a theoretical result for a single operation. **Resolution**: For mixed operations (add + sub + mult) and practical training stability, 2-4 layers is the pragmatic choice. 1 layer is a theoretical lower bound, not an engineering recommendation.

## Counter-arguments

- **Deeper models may help with harder math**: These findings are specific to basic arithmetic (add/sub/mult). Algebra, calculus, or mathematical reasoning may require genuine depth. **Disproof condition**: If our model later needs to handle more complex operations and shallow architectures fail, this theme needs revision.
- **Looped transformers add implementation complexity**: The weight-tied loop with input injection and progressive loss is non-trivial to implement. Standard frameworks don't support it natively. **Disproof condition**: If the standard 4-layer model achieves our accuracy targets without loops, the complexity isn't justified.

## So What?

**For our architecture spec**: Build a 4-layer, 4-head, 256d model as the primary configuration (~3-5M params). Use RMSNorm pre-normalization. If multiplication accuracy is insufficient, try a 2-layer x 4-loop configuration before adding more unique layers. The parameter budget ceiling of 100M is far above what's needed — even 10M is generous.
