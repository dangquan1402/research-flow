---
title: "Finding: Scratchpad Training Dramatically Improves Arithmetic but at Significant Sequence Length Cost"
created: 2026-04-13
updated: 2026-04-13
source: training-curriculum-research
confidence: high
verification: source
tags: [training, arithmetic, scratchpad, chain-of-thought]
related: [data-format-impact, length-generalization-techniques]
staleness_days: 0
---

## Insight

Training transformers to emit intermediate computation steps (scratchpad / chain-of-thought) dramatically improves accuracy, sample efficiency, and convergence speed on multi-step arithmetic, even when training from random initialization without pretraining. However, this comes at a 3-10x increase in sequence length, which increases compute and memory costs. The simplified scratchpad (digit-wise sum + carry) offers the best accuracy/cost trade-off.

## Evidence

- Nye et al. (2022) "Show Your Work" demonstrated that scratchpad training dramatically improves multi-step computation by allowing models to emit and condition on intermediate tokens.
- Lee et al. (2023) showed four format levels: Plain < Reverse < Simplified Scratchpad < Detailed Scratchpad, with each level improving accuracy and sample efficiency. Importantly, scratchpad works even without pretraining.
- Feng et al. (2023) proved theoretically that chain-of-thought with linear steps keeps transformer decoders within context-sensitive languages, while polynomial steps enable recognition of all polynomial-time solvable problems.
- Bartan et al. (2024) identified the "globality barrier" -- a fundamental limitation of standard scratchpad on certain tasks -- and proposed "inductive scratchpad" to overcome it.
- The scratchpad format effectively converts a hard single-step prediction into a sequence of easy single-step predictions, each conditioned on the previous.

## Reasoning

Without scratchpad, a transformer must compute multi-digit arithmetic in a single forward pass. The number of "reasoning steps" available is limited by model depth (number of layers). For complex carry propagation or multi-digit multiplication, this may be insufficient. Scratchpad externalizes the computation into the sequence, allowing the model to use its autoregressive decoding loop as an unbounded computation mechanism. Each generated intermediate token adds one more "step" of computation.

The simplified scratchpad (carry + digit sum) is particularly effective because it encodes exactly the information the model needs without redundancy. The detailed scratchpad adds more tokens but provides diminishing returns since the additional verbosity doesn't carry new information.

## Counter-arguments

- Sequence length increase (3-10x) directly multiplies training FLOPS due to quadratic attention cost.
- At inference time, scratchpad increases latency proportionally to the number of intermediate tokens.
- Reverse format achieves competitive accuracy without the sequence length overhead, making it potentially more practical.
- The model must learn to generate correct intermediate steps, which is itself a learning challenge -- if intermediate steps are wrong, the final answer will also be wrong (error propagation).
- Scratchpad format is specific to arithmetic and doesn't easily compose with natural language tasks.

## Implications

- **For our project**: Start with reverse format as the default. Implement simplified scratchpad as an option for cases where reverse format is insufficient (e.g., multi-digit multiplication).
- If using scratchpad, mask the loss on the input portion of the sequence to focus learning on the computation steps.
- Budget 3-5x more tokens per training example when using scratchpad format.
- Consider whether inference latency is acceptable for the target use case before committing to scratchpad.
- The "inductive scratchpad" from Bartan et al. (2024) deserves further investigation for our hardest tasks.
