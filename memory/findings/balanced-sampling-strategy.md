---
title: "Finding: Balanced Carry Sampling Is Critical for Training Data Quality"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, sampling, data-distribution]
related: [data-format-impact, length-generalization-techniques, curriculum-learning-arithmetic]
staleness_days: 0
---

## Insight

Uniform random sampling of operands produces severely skewed training data: 90% of numbers in [0, 999] have 3 digits, and long carry chains are exponentially rare. Balanced carry sampling -- uniformly sampling the carry chain length first, then sampling operands that produce that carry length -- is essential for strong generalization, particularly for length generalization on addition. Balanced digit sampling (upweighting shorter numbers) ensures coverage across all digit lengths.

## Evidence

- When sampling operands randomly for addition, the dataset is heavily skewed in both digit count distribution and carry operation frequency. Long carry chains (e.g., 999+1=1000) are extremely rare but critical for the model to learn.
- Balanced carry sampling first samples carry chain length uniformly between 0 and the question length, then samples random operands producing that carry length. This ensures the model sees significant numbers of long carry chain examples.
- This technique enabled the first instance of strong length generalization on decimal addition for transformers trained from scratch (McLeish et al., 2024).
- Lee et al. (2023) showed that structured sampling balancing different carry operation types is important for learning, since certain carry patterns are severely underrepresented in uniform random data.
- Balanced digit sampling assigns higher weights to operands with fewer digits (e.g., 7 is sampled more often than 700 to offset the natural distribution).

## Reasoning

The core problem is that uniform sampling over integers is not uniform sampling over difficulty or over the skills needed. A model trained predominantly on no-carry or short-carry additions will fail on long-carry examples at test time. Since carry propagation is the core difficulty of addition (and the primary source of length generalization failure), ensuring the model sees diverse carry patterns is directly addressing the bottleneck.

This is analogous to class-balanced sampling in classification: the rare but important cases must be oversampled to prevent the model from ignoring them.

## Counter-arguments

- Balanced carry sampling requires computing the carry chain for each example, adding preprocessing cost.
- Over-representing rare carry patterns could distort the model's implicit understanding of the natural distribution of numbers (though this may not matter for a pure arithmetic model).
- For operations other than addition (e.g., multiplication), the analogous "difficulty-balanced" sampling is less well-defined.

## Implications

- **For our project**: Implement balanced carry sampling as a core part of the data pipeline. This is non-negotiable for addition training.
- Also implement balanced digit sampling to ensure coverage across all operand lengths.
- For multiplication, investigate analogous difficulty-balanced sampling (e.g., balancing on the number of partial product carries or output length).
- Log the carry distribution and digit distribution of generated training data as a sanity check.
- This is a data-side technique with zero architectural cost -- implement it regardless of other design choices.
