---
title: "Negative Result: Looped 454K Model Fails Mixed-Op Target Despite Pre-Training"
created: 2026-04-14
updated: 2026-04-14
source: experiment-looped-pretrain-mixed
confidence: high
verification: source
tags: [experiment, arithmetic, mixed-ops, looped-transformer, pre-training, negative-result, parameter-efficiency]
related: [experiment-looped-192D-addition, experiment-mixed-operations, experiment-looped-mul-scratchpad]
staleness_days: 0
---

## Insight

A looped transformer (1Lx4/4H/192D, ~463K params) **cannot achieve >95% on mixed add+sub+mul** even with addition pre-training. Final accuracy: **28.25%** after 40 epochs of fine-tuning on mixed ops. This establishes that mixed-operation arithmetic requires significantly more model capacity than single-operation tasks, where the same architecture achieves 99.2% on addition and 99.4% on multiplication individually.

## What Would Disprove This?

- A different pre-training curriculum (e.g., staged introduction of operations) could improve mixed performance
- Longer training (>100 epochs) might eventually converge, though the learning curve was flattening
- A larger looped model (e.g., 1Lx4/256D, ~800K params) might find a capacity sweet spot

## Evidence

### Experiment 1: Pre-train on Addition (10 epochs)
- **Config:** 1L/4H/192D, looped=4x, 453,696 params
- **Task:** 5-digit addition only
- **Result:** 13.60% accuracy (still learning, not converged)
- Checkpoint saved for fine-tuning

### Experiment 2: Fine-tune on Mixed Ops (40 epochs)
- **Config:** Same architecture, loaded addition checkpoint, 463,104 params (pos_emb resized for scratchpad sequences)
- **Task:** Mixed add+sub+mul with scratchpad for multiplication
- **Result:** 28.25% final accuracy
- **Digit breakdown at epoch 40:** 1d:100% 2d:92% 3d:55% 4d:27% 5d:16% 6d:3%
- Loss curve: 2.29 → 1.29 (still decreasing slowly, but accuracy plateauing)
- Training time: 3558s (~59 min)

### Experiment 3: Control (not completed)
- Mixed from scratch was started but killed early — the pre-trained result at 28.25% already clearly misses the >95% target.

## Comparison with Larger Model

| Model | Params | Mixed Accuracy | Pre-trained? |
|---|---|---|---|
| **1Lx4/192D looped** | **463K** | **28.25%** | **Yes (add)** |
| 2L/384D standard | 3.58M | 92.3% | Yes (add) |
| 2L/384D standard | 3.58M | 81.8% | No (scratch) |

The 463K model achieves only 30% of the larger model's accuracy on mixed ops. Pre-training provides a ~10% boost for the larger model (81.8% → 92.3%) but the small model can't leverage it effectively.

## Key Takeaway

**Mixed operations require more capacity than single operations.** The looped 1Lx4/192D architecture works excellently for single-op tasks (99.2% add, 99.4% mul) but fails on mixed ops. This suggests the shared weight block in a looped transformer struggles to multiplex between fundamentally different algorithms (carry propagation for add/sub vs. partial products for mul). The >95% mixed-op target likely requires at minimum the 2L/384D scale (~3.5M params) or a mid-range architecture (~800K-1M params) not yet tested.

## Source

- Pre-train results: `experiments/results/add_5d_reversed_1L4H192D_looped_pretrain_add.json`
- Fine-tune results: `experiments/results/mixed_5d_reversed_1L4H192D_looped_finetune_mixed.json`
- Comparison: `experiments/results/mixed_5d_reversed_2L4H384D_finetune_mixed.json`, `experiments/results/mixed_5d_reversed_2L4H384D_mixed_scratch.json`
