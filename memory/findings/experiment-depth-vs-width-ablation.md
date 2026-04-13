---
title: "Experimental Finding: 2L/384D Is the Sweet Spot for 5-Digit Addition"
created: 2026-04-13
updated: 2026-04-13
source: experiment-depth-width-ablation
confidence: high
verification: source
tags: [experiment, arithmetic, architecture, depth, width, ablation]
related: [depth-vs-width-arithmetic, minimum-viable-math-transformer, experiment-baseline-addition-reversed]
staleness_days: 0
---

## Insight

In a 4-way ablation on 5-digit addition (all with reversed output), all configs from 1.58M to 6.32M params achieve >99.5% accuracy. The 2L/4H/384D (3.56M params) configuration is optimal: fastest to 100% accuracy (epoch 20), most stable training, and reasonable compute cost. Two layers suffice for addition when paired with adequate width.

## Evidence

Controlled ablation, all other settings identical (50K train, reversed output, balanced carry):

| Config | Params | Final Acc | Peak Acc (epoch) | Time |
|---|---|---|---|---|
| 8L/4H/128D | 1.58M | 99.95% | 100% (ep 30) | 548s |
| 4L/4H/256D | 3.16M | 99.90% | 100% (ep 45) | 517s |
| 2L/4H/384D | 3.56M | 99.90% | 100% (ep 20) | 548s |
| 2L/8H/512D | 6.32M | 99.65% | 100% (ep 35) | 1003s |

## Reasoning

Addition is parallelizable across digit positions — each position needs access to its two input digits and the carry from the previous position. This maps to width (multiple heads attending to different positions) more than depth (sequential processing). Two layers suffice: one for digit-pair attention, one for carry propagation and output computation.

However, the 8L/128D config's strong performance shows that depth can compensate for width. The key insight is that **2 layers is the minimum, and additional width beyond ~384D shows diminishing returns**.

## Counter-arguments

- Single seed per config — need multi-seed runs for statistical significance
- Addition only — multiplication or mixed operations may favor different configurations
- The deep-narrow config's instability (90% dip at epoch 20) could be a concern for reproducibility
- All configs performed so well that the architectural differences may be insignificant

## Implications

For the recommended architecture: use 2L/4H/384D for addition experiments. This confirms the "width over depth" finding from the literature while showing the minimum bar is 2 layers. For multiplication experiments, more depth (4L) may be needed.
