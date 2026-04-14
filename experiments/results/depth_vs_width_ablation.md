# Depth vs Width Ablation: 5-Digit Addition

## Experiment Design

All configurations use reversed output, balanced carry sampling, same hyperparameters (AdamW lr=1e-3, wd=0.5, cosine schedule, 50K train, 2K test, 50 epochs). Only architecture varies.

## Results Summary

| Config | Params | Final Acc | Peak Acc (epoch) | Epoch@95% | Total Time |
|---|---|---|---|---|---|
| 8L/4H/128D (deep-narrow) | 1.58M | 99.95% | 100% (ep 30) | ~10 | 548s |
| 4L/4H/256D (balanced) | 3.16M | 99.90% | 100% (ep 45) | ~15 | 517s |
| 2L/4H/384D (shallow-wide) | 3.56M | 99.90% | 100% (ep 20) | ~10 | 548s |
| 2L/8H/512D (very wide) | 6.32M | 99.65% | 100% (ep 35) | ~10 | 1003s |

## Detailed Analysis

### Deep-narrow (8L/4H/128D, 1.58M params)
- Surprisingly strong: 99.95% with the fewest parameters
- Reaches 98.35% by epoch 10 — fastest to high accuracy
- Some instability mid-training (90.05% at epoch 20, recovered)
- Shows more variance between evaluation points

### Balanced (4L/4H/256D, 3.16M params)
- Consistent, stable training progression
- Reaches 95% by epoch 15
- Smooth accuracy curve with less variance
- Best training speed (517s, ~8.6 min)

### Shallow-wide (2L/4H/384D, 3.56M params)
- Reaches 100% by epoch 20 — fastest to perfect
- Extremely stable: 99.85%+ from epoch 15 onward
- Loss continues decreasing (1.2846 at epoch 50 vs ~1.30 for others)
- Comparable training time to balanced config

### Very wide (2L/8H/512D, 6.32M params)
- 2x the parameters, no accuracy improvement
- Slower training (1003s vs 548s) due to larger matrices
- Reaches 100% at epoch 35 but with more fluctuation
- Loss drops further (1.1594) suggesting potential overfitting

## Key Findings

1. **All configurations achieve >99.5% accuracy** — for 5-digit addition with reversed output, architecture matters less than data formatting.

2. **Shallow-wide (2L/384D) is the sweet spot** — fastest to 100%, most stable, and comparable training cost to the balanced config.

3. **Deep-narrow (8L/128D) is surprisingly competitive** — achieves 99.95% with only 1.58M params, but shows more training instability.

4. **Diminishing returns beyond ~3.5M params** — the 6.32M model is slower without being more accurate.

5. **Width > depth confirmed for addition** — both 2L configs reach high accuracy quickly, while the balanced 4L config is slightly slower to converge.

6. **The research findings about depth vs width are nuanced** — contrary to the strong "width dominates" claim, depth also works well with small enough width. The real finding is that **2 layers is sufficient for addition** when paired with adequate width.

## Recommendation

For 5-digit addition: **2L/4H/384D** (~3.5M params) — fastest to perfect accuracy, stable training, reasonable compute cost. If minimizing parameters is important, 8L/4H/128D (~1.6M) works but with less stability.
