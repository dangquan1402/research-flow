---
title: "Finding: Minimum Viable Architecture for >95% Arithmetic Accuracy"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
confidence: high
verification: source
tags: [architecture, arithmetic, minimum-viable, parameter-budget]
related: [depth-vs-width-arithmetic, attention-heads-carry-operations, positional-encoding-arithmetic]
staleness_days: 0
---

## Insight

A transformer with 2 layers, 3-4 attention heads, and ~10M parameters is the minimum viable architecture for >99% accuracy on multi-digit addition and subtraction. For multiplication, 2-3 layers with auxiliary losses or progressive training are needed. The total parameter budget for a capable arithmetic transformer is 10-15M — well within consumer hardware constraints. Data formatting and positional encoding choices matter more than raw architecture size.

## Evidence

1. **Addition — 2 layers, 3 heads (Nikankin 2024):** >99.999% accuracy on n-digit addition. The 3 heads specialize in base addition, carry detection, and carry cascading. This is the smallest known configuration for near-perfect addition.

2. **Addition — 6 layers, 6 heads, 384 embed dim, 10.6M params (Lee 2023):** NanoGPT achieves perfect addition with reversed output format. The architecture is likely overprovisioned — the key was data formatting (reverse LSB-first), not model size.

3. **Addition with generalization — 1 layer, 4 heads (Cho 2024):** With position coupling, even a 1-layer transformer generalizes to 200-digit addition (6.67x training length). Theoretical proof that 1 layer + coupled positions suffices for exponentially long addition.

4. **Multiplication — tiny transformer, 3 heads (2024):** >99.9% accuracy on 5-digit multiplication with reversed format + progressive training. Outperforms GPT-4.

5. **Multiplication with auxiliary losses — 2 layers (2025):** 99% on 4x4 multiplication by adding auxiliary loss terms for intermediate sums. Raw 12-layer models without this fail entirely.

6. **Grokking — 2 layers, 4 heads, 128 dim (Power 2022):** Standard configuration for modular arithmetic grokking experiments. Even 2 layers / 1 head / 128 dim achieves grokking in <150 epochs.

7. **Training compute:** 10M-param models train under 1 hour on single consumer GPU. 85K-param nanoGPT trains in ~3 minutes on CPU.

## Reasoning

The minimum architecture is driven by the computational structure of arithmetic:

- **Addition:** Each digit position requires (1) accessing the two input digits, (2) computing their sum, (3) determining the carry. This needs at minimum 2-3 attention heads (one per subtask) and 1-2 layers (one for digit-pair attention, one for carry propagation). The MLP layers combine head outputs.

- **Subtraction:** Structurally similar to addition (borrow instead of carry), same minimal architecture applies.

- **Multiplication:** Requires computing partial products and accumulating them, which involves more intermediate steps. 2-3 layers help, but the critical factor is auxiliary training signals, not raw depth.

The 10M parameter sweet spot arises from: ~384 embed dim * 6 heads * 6 layers with character-level vocabulary. But evidence suggests this is overprovisioned — 2-4 layers with 256-384 embed dim and 3-6 heads (1-5M params) likely suffices.

## Counter-arguments

- **Generalization vs accuracy:** Minimal architectures achieve high accuracy on trained digit lengths but may not generalize to longer inputs. Length generalization requires specialized positional encodings (Position Coupling, Abacus) regardless of model size.
- **Mixed operations:** Training on addition + subtraction + multiplication simultaneously may require more capacity than any single operation. Published work on mixed models used 2-3 layers with 3-4 heads, but comprehensive mixed-operation benchmarks are limited.
- **Robustness:** Minimal architectures may be brittle — achieving 99.999% on clean data doesn't guarantee robustness to noisy or adversarial inputs.

## Implications

Recommended architecture for our project (addition + subtraction + multiplication, <100M params):

**Conservative (high confidence):**
- 4 layers, 4 heads, 384 embed dim, ~10M params
- Character-level tokenization with reverse output format
- Pre-norm (RMSNorm or LayerNorm)
- FFN multiplier 4x (1536 FFN dim)
- Trains in <1 hour on single consumer GPU

**Aggressive (medium confidence):**
- 2 layers, 4 heads, 256 embed dim, ~2-3M params
- Same tokenization/formatting
- May need auxiliary losses for multiplication
- Trains in minutes on consumer GPU

**With length generalization:**
- Add Position Coupling or Abacus Embeddings to either config
- Looped transformer variant (2 layers x 4 loops) for parameter-efficient depth

The parameter budget ceiling of 100M is far above what's needed. Even 10M is generous for this task.
