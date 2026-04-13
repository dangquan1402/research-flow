---
title: "Finding: Data Format Is the Single Most Important Training Decision"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, data-format]
related: [curriculum-learning-arithmetic, length-generalization-techniques]
staleness_days: 0
---

## Insight

The format in which arithmetic problems are presented to the model has the largest single impact on accuracy, sample efficiency, and convergence speed. Plain left-to-right format (e.g., "123+456=579") never reaches 100% accuracy on multi-digit addition, while reverse (Little-Endian) output or scratchpad formats achieve near-perfect accuracy with far fewer training examples.

## Evidence

- Lee et al. (2023) compared four formats: Plain, Reverse, Simplified Scratchpad, and Detailed Scratchpad. Plain format never reached 100% accuracy. Each additional level of detail in the format improved both accuracy and sample efficiency.
- Shen et al. (2024) "RevOrder" achieved 100% accuracy on addition, subtraction, multiplication, and low-digit division using reversed output format, outperforming baselines while using only 5.2% of the training tokens.
- Gao et al. (2024) "Reverse That Number!" improved accuracy by 11.1% over state-of-the-art for large-digit inputs using Little-Endian format.
- The fundamental issue: in plain format, the model must predict the most significant digit first, before it has generated (and thus can condition on) the carry information from less significant digits. This creates an impossible prediction problem for the autoregressive model.

## Reasoning

Autoregressive models generate tokens left-to-right. For addition, carry propagation flows right-to-left (least to most significant digit). Plain format creates a fundamental mismatch: the first output token requires information that won't be available until the last token is generated. Reverse format aligns generation order with carry flow. Scratchpad formats explicitly externalize the carry state so it's available when needed.

This is not merely a training efficiency issue -- it's an architectural constraint. Without reverse or scratchpad format, the model must learn to internally simulate carry propagation across all digits simultaneously in a single forward pass, which requires significant model capacity and may be impossible for multi-digit problems with a small model.

## Counter-arguments

- Very large pretrained LLMs can sometimes do plain-format arithmetic through in-context learning, suggesting that with enough parameters and pretraining data, the format constraint can be partially overcome.
- Scratchpad formats significantly increase sequence length (3-10x), which increases training compute and memory.
- Reverse format is non-standard and may complicate integration with natural language tasks in a multi-task model.

## Implications

- **Critical decision for our project**: Use reverse (Little-Endian) format as the default for our arithmetic LLM. It achieves the best accuracy-to-compute ratio.
- If accuracy on hard cases is paramount and compute budget allows, use simplified scratchpad format.
- Avoid plain format entirely for any serious arithmetic training.
- Consider: if we later want to integrate arithmetic into a general-purpose model, we may need a format translation layer.
