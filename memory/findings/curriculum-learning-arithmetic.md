---
title: "Finding: Curriculum Learning Helps Sample Efficiency but Is Not Strictly Required"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: medium
verification: source
tags: [training, arithmetic, curriculum]
related: [data-format-impact, length-generalization-techniques]
staleness_days: 0
---

## Insight

Curriculum learning (progressive difficulty ordering) improves sample efficiency for arithmetic transformers but is neither necessary nor sufficient. Data format and sampling strategy matter more. When combined with good data formatting (reverse output or scratchpad) and balanced carry sampling, curriculum learning provides only marginal additional benefit.

## Evidence

- Lee et al. (2023) "Teaching Arithmetic to Small Transformers" showed a progressive digit curriculum (1-digit -> 2-digit -> 3-digit) works: after 96% accuracy on 1-digit, switching to 2-digit drops accuracy to ~5%, then recovers to 90%+. However, the same paper showed that data format changes (reverse, scratchpad) had a much larger effect than curriculum ordering.
- Chen et al. (2025) introduced "position curriculum" -- broadening the range of absolute character positions during training -- which improves robustness to input-format variation for modular arithmetic.
- Liu et al. (2025) "Shattered Compositionality" found counterintuitive learning dynamics where transformers learn component skills in unexpected orders, suggesting curriculum design based on human intuition about difficulty may not match the model's actual learning trajectory.

## Reasoning

The autoregressive loss already provides implicit curriculum pressure: the model learns easier patterns (single-digit, no carry) first because they have lower loss. Explicit curriculum adds structure on top of this but can also restrict the training distribution in harmful ways (e.g., if the model overfits 1-digit patterns before seeing multi-digit data). The evidence suggests that ensuring diverse, well-balanced training data at all difficulty levels simultaneously is more robust than strict curriculum ordering.

## Counter-arguments

- For very resource-constrained training (limited compute budget), curriculum learning may still be the fastest path to acceptable accuracy on a specific digit range.
- Position curriculum (as opposed to difficulty curriculum) has stronger evidence of benefit, particularly for robustness.
- Some practitioners report curriculum learning is essential when training from scratch without pre-training, though this conflicts with Lee et al.'s findings.

## Implications

- For our arithmetic LLM: implement curriculum learning as an optional mode but invest more effort in data format and balanced sampling.
- If using curriculum, prefer a "mixed" approach: train on all difficulty levels simultaneously but with higher weight on the current target difficulty, rather than strict sequential staging.
- Position curriculum (varying where expressions appear in the input) is likely more valuable than difficulty curriculum.
