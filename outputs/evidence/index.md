---
title: "Evidence Index"
created: 2026-04-15
updated: 2026-04-15
goal: math-llm-architecture
---

## Evidence Artifacts

| # | Claim | Findings Backed | Type | File | Status |
|---|-------|----------------|------|------|--------|
| 1 | Reversed (LSB-first) output learns 2x faster and +2% more accurate than plain | reversed-digit-order, experiment-tokenization-comparison | chart (2-panel: learning curves + per-digit accuracy) | [reversed_vs_plain.png](reversed_vs_plain.png) | generated |
| 2 | Width > depth; 2L/384D is sweet spot for addition | depth-vs-width-arithmetic, experiment-depth-vs-width-ablation | chart (2-panel: convergence + params vs accuracy) | [depth_vs_width.png](depth_vs_width.png) | generated |
| 3 | Scratchpad boosts multiplication from 85-95% to 100% | experiment-scratchpad-2L384D, experiment-scratchpad-4L256D | chart (2-panel: learning curves + before/after bars) | [scratchpad_impact.png](scratchpad_impact.png) | generated |
| 4 | Looped transformers achieve near-baseline accuracy with 87% fewer params | experiment-looped-192D-addition, experiment-looped-mul-scratchpad | chart (2-panel: scatter + comparison bars) | [looped_efficiency.png](looped_efficiency.png) | generated |
| 5 | SwiGLU converges 5 epochs faster than GELU at equal param count | experiment-swiglu-fair-comparison | chart (2-panel: accuracy + loss curves) | [activation_comparison.png](activation_comparison.png) | generated |
| 6 | Single model handles add+sub+mul with no catastrophic interference | experiment-mixed-operations | chart (2-panel: per-op learning curves + final bars) | [mixed_ops_training.png](mixed_ops_training.png) | generated |
| 7 | APE fails completely on OOD lengths; Position Coupling shows weak signal | experiment-length-generalization, experiment-position-coupling-vs-ape | chart (2-panel: OOD accuracy bars + convergence) | [length_generalization.png](length_generalization.png) | generated |
| 8 | Multiplication difficulty scales steeply with digit count | experiment-scratchpad-2L384D, multiplication-scaling-requirements | chart (2-panel: 3d convergence + 3d vs 5d scaling) | [multiplication_scaling.png](multiplication_scaling.png) | generated |
| 9 | Master summary of all 22 experiments | all findings | color-coded table (green/yellow/orange/red by accuracy) | [master_summary.png](master_summary.png) | generated |

## Scripts

All evidence is reproducible from experiment JSON data via Python scripts in this directory:

| Script | Generates |
|--------|-----------|
| `reversed_vs_plain.py` | reversed_vs_plain.png |
| `depth_vs_width.py` | depth_vs_width.png |
| `scratchpad_impact.py` | scratchpad_impact.png |
| `looped_efficiency.py` | looped_efficiency.png |
| `activation_comparison.py` | activation_comparison.png |
| `mixed_ops_training.py` | mixed_ops_training.png |
| `length_generalization.py` | length_generalization.png |
| `multiplication_scaling.py` | multiplication_scaling.png |
| `master_summary.py` | master_summary.png |
