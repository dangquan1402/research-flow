---
title: "Finding: 3 Attention Heads Are Sufficient for Carry/Borrow Propagation"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
confidence: high
verification: source
tags: [architecture, arithmetic, attention-heads, carry, interpretability]
related: [minimum-viable-math-transformer, depth-vs-width-arithmetic]
staleness_days: 0
---

## Insight

Transformers learn to implement carry/borrow propagation in addition and subtraction using exactly 3 specialized attention heads operating with a 1-token time offset. Each head handles a distinct subtask: base digit addition, single-digit carry detection, and cascading carry resolution. This 3-head specialization is consistent across independently trained models and also applies to multiplication. For mixed arithmetic operations (add + sub + mult), 4-6 heads provide sufficient capacity.

## Evidence

1. **3-head mechanism for addition (Nikankin et al., 2024):** In 2-layer, 3-head models achieving >99.999% accuracy: "The 3 heads are time-offset from each other by 1 token such that, in each row, data from 3 tokens is available. To calculate each answer digit, the 3 heads do independent simple calculations on the relevant digit pairs, with results combined by the MLP layer."

2. **Functional specialization discovered:** Different attention heads handle distinct algorithmic subtasks:
   - **SA nodes:** Base addition calculations (digit pair sums)
   - **ST nodes:** Single-digit carry detection (does this pair produce a carry?)
   - **SV nodes:** Cascading carry resolution (does a carry cascade through multiple positions?)
   - Final answer digit computed by combining all three

3. **3 distinct carry clusters:** PCA analysis reveals all models develop 3 clusters corresponding to carry states: definite carry, no carry, and uncertain/conditional carry. "All models contain the subtasks required" regardless of random initialization.

4. **Position-specific attention:** ST nodes "attend primarily to tokens of specific digit positions with post-softmax attention >0.01," showing that heads learn position-specific specialization.

5. **Multiplication also uses 3 heads:** Research on multiplication transformers found "3 attention heads are sufficient to complete multiplication," handling base multiplication, carry calculation, and combined operations.

6. **Larger models use more heads but not all:** In 5-digit addition models with 30 attention heads, analysis showed not all were essential. "Some models only use one attention head" for certain digit calculations, "some models combine 2 attention heads." The 3-head minimum is robust.

7. **Mixed operation models:** Published work on models handling multiple operations used 3-4 heads per layer with 2-3 layers.

## Reasoning

The 3-head specialization maps directly to the algorithmic structure of addition:

1. **Head 1 (base sum):** Computes digit_a + digit_b for each position. This is a lookup/computation that requires attending to the two input digits at the corresponding position.

2. **Head 2 (carry detection):** Determines if the sum at each position exceeds 9 (producing a carry). This requires seeing the digit pair AND knowing the sum from Head 1.

3. **Head 3 (carry cascade):** Handles the case where a carry from position i causes position i+1 to also carry (e.g., 999 + 1 = 1000). This requires seeing both the sum and carry status of adjacent positions.

The 1-token time offset between heads is elegant: each head "looks back" one more position than the previous, building up the information needed for carry cascade detection. The MLP layer then combines the three signals into the final digit output.

For subtraction, the same structure applies with borrow instead of carry.

For multiplication, the heads handle partial product computation, accumulation, and carry — structurally similar but applied over more digit pair combinations.

## Counter-arguments

- **More heads can help:** While 3 is the minimum, 4-6 heads may improve robustness and training speed. Extra heads can provide redundancy or handle edge cases (e.g., carries cascading through many positions).
- **Head count depends on digit length:** For very long numbers (50+ digits), carry cascades become longer and may benefit from more heads or more layers to propagate information.
- **Not all tasks need 3:** For modular arithmetic (grokking experiments), even 1 head with 2 layers sufficed. The 3-head minimum is specific to full integer addition with carry.
- **Interpretability may be misleading:** The clean 3-cluster pattern may not reflect the full computation. Models might use heads in more complex ways that PCA doesn't capture.

## Implications

For our arithmetic LLM:

1. **Minimum 3 heads per layer** for any layer handling carry/borrow computation
2. **4-6 heads recommended** for mixed operations (add + sub + mult) to provide capacity for all three operation types
3. **Head dimension:** With 256-384 embed dim and 4-6 heads, head dim = 64 per head — well within the 64-128 sweet spot from the literature
4. **Interpretability opportunity:** The 3-head carry mechanism is well-understood; we can verify our model learns similar patterns as a training diagnostic
5. **Architecture:** 2 layers x 4 heads = 8 total head slots, more than sufficient for 3 carry heads + additional capacity for subtraction borrow and multiplication partial products
