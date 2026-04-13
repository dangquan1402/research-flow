---
title: "Finding: AdamW with High Weight Decay and Cosine Schedule Is the Standard Recipe"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, optimizer, learning-rate]
related: [grokking-arithmetic]
staleness_days: 0
---

## Insight

AdamW with weight decay 0.1-1.0, learning rate 1e-3, beta2=0.99, linear warmup (100 steps), and cosine decay is the established recipe for training small arithmetic transformers. The weight decay parameter is unusually important because it directly controls grokking dynamics and generalization. Batch size 256 is a reasonable default for ~10M parameter models.

## Evidence

- Lee et al. (2023) "Teaching Arithmetic to Small Transformers" used: AdamW, lr=1e-3, beta2=0.99, batch size 256, 100 warmup iterations, 10K max iterations, on a 10.6M parameter NanoGPT (6 layers, 6 heads, embedding dim 384).
- Power et al. (2022) and Nanda et al. (2023) showed that weight decay is the single most impactful hyperparameter for grokking, with higher weight decay (0.1-1.0) accelerating generalization. Weight decay more than halved the number of samples needed compared to other interventions.
- Cosine decay with warmup is the standard schedule across transformer training literature. The warmup phase prevents early training instability; cosine decay provides smooth annealing.
- Recent work on component-specific learning rates shows that tuning LR separately for embedding, attention, and FFN layers can improve performance, and these relative rates transfer from small to large models.

## Reasoning

AdamW decouples weight decay from the gradient update, which is important for arithmetic tasks because:
1. Weight decay needs to act uniformly on all parameters to discourage memorization
2. With coupled weight decay (as in L2-regularized Adam), the effective regularization varies with gradient magnitude, creating inconsistent pressure
3. For grokking, consistent weight decay pressure is what drives the transition from memorization to generalization

The lr=1e-3 value is standard for small transformers. Beta2=0.99 (vs. the default 0.999) allows faster adaptation of the second moment estimate, which may help with the sharp loss landscape of arithmetic tasks.

## Counter-arguments

- Some recent work shows SGD with proper LR tuning can match Adam at very small batch sizes, though this is not standard practice.
- The optimal weight decay for grokking may not be optimal for standard generalization -- there may be a trade-off between grokking speed and final performance.
- For very long training runs (needed for grokking), cosine decay may need to be replaced with constant LR + late decay, since cosine reaches near-zero LR before grokking occurs.

## Implications

- **For our project**: Use AdamW with lr=1e-3, beta2=0.99, weight_decay=0.5 as the starting point. Tune weight_decay in the range [0.1, 1.0].
- Use linear warmup for 100 steps, then cosine decay.
- Batch size 256 is a good default. If GPU memory allows, don't go much higher -- larger batches can hurt generalization on small datasets.
- For grokking experiments: consider constant LR (no decay) or very slow decay to maintain gradient signal throughout the long training.
- Monitor the ratio of weight norm to gradient norm as a diagnostic for grokking progress.
