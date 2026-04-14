---
title: "Tokenization Strategies for Arithmetic LLMs"
created: 2026-04-13
type: web-search
ingested_by: "hypothesis/GH-2-tokenization-strategies"
---

# Tokenization Strategies for Arithmetic LLMs

## Overview

Tokenization is widely regarded as the single most important design decision when training small LLMs for arithmetic. The way numbers are represented as tokens determines whether a model can learn carry propagation, generalize to longer digit sequences, and achieve high accuracy on basic operations (addition, subtraction, multiplication).

This document synthesizes findings from approximately a dozen papers published 2021--2025 investigating how tokenization and data formatting affect arithmetic learning in transformers.

---

## 1. Digit-Level (Character-Level) Tokenization

**Core idea:** Each digit and operator is a separate token. E.g., `3 + 1 5 7 = 1 6 0`.

### Key Papers
- **Lee et al. (2024)** "Teaching Arithmetic to Small Transformers" (ICLR 2024) -- uses character-level tokenization with NanoGPT (10.6M params). Achieves 100% accuracy on multi-digit addition with reversed output and ~2,500 training samples.
- **Stolfo et al. (2024)** "Arithmetic in Transformers Explained" -- each digit as a separate token; achieves 99.999% accuracy on 5--15 digit problems with 2--3 layer models.
- **Nogueira et al. (2021)** "Investigating the Limitations of Transformers with Simple Arithmetic Tasks" -- demonstrated that character-level representation struggles on 5+ digit addition unless augmented with position tokens.

### Why It Works
- Each digit occupies exactly one token position, creating a natural alignment between operand digits and output digits.
- The model can learn per-position operations (digit-pair addition, carry detection) as separable subtasks.
- Attention heads can attend to the corresponding digit positions in each operand.

### Limitations
- Does not inherently solve the carry propagation direction problem (see Section 2).
- Vocabulary is tiny (0--9, +, -, *, =, padding tokens), which may under-utilize model capacity.

---

## 2. Reversed Digit Order (Least-Significant Digit First)

**Core idea:** Output the answer starting from the least-significant digit. E.g., `1 2 3 + 4 5 6 = 9 7 5` (reversed: 579 = 975 reversed is 579... example: `123 + 456 = 975` becomes output `5 7 9`).

### Key Papers
- **Lee et al. (2024)** -- reversed output achieves 100% accuracy with ~2,500 samples vs. 85% plateau for standard (MSB-first) output with 10,000 samples.
- **"Reverse That Number! Decoding Order Matters in Arithmetic Learning"** (2024, arXiv:2403.05845) -- proposes LEFT (Little-Endian Fine-Tuning). Achieves 11.1% overall accuracy improvement over previous SOTA using only 5.2% of training tokens for addition/subtraction.
- **"RevOrder: A Novel Method for Enhanced Arithmetic"** (Shen, 2025) -- 100% accuracy on addition, subtraction, multiplication with reversed output on Big-bench arithmetic.

### Why It Works
- Standard addition by humans proceeds right-to-left (least-significant digit first). Autoregressive transformers generate tokens left-to-right. Reversing the output aligns the generation order with the natural computation order.
- **Carry propagation becomes causal:** When generating the i-th output digit (which represents the i-th least-significant position), the carry from position i-1 has already been computed and is available in the context.
- **Computational complexity drops from exponential to linear:** LEFT paper proves standard (Big-Endian) learning complexity is C_Big >= 10^(2n+2) while Little-Endian is C_Little <= n * 10^5.
- **Robustness:** Under noisy intermediate token conditions, reversed format maintains 81.26% exact accuracy vs. 49.88% for plain format (Lee et al.).

### Specific Results (LEFT paper)
| Operation | LEFT Accuracy | Prior SOTA | 
|-----------|--------------|------------|
| Addition | 98.8% | 99.8% |
| Subtraction | 95.9% | 97.3% |
| Multiplication | 88.5% | 52.8% |
| **Overall** | **94.4%** | **83.3%** |

Training required only ~3M tokens vs. ~11M for prior SOTA.

---

## 3. BPE / Subword Tokenization

**Core idea:** Standard LLM tokenizers (BPE, SentencePiece) merge frequent byte/character sequences into subword tokens.

### Key Paper
- **"Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs"** (2024, arXiv:2402.14903)

### Why BPE Hurts Arithmetic
1. **Inconsistent segmentation:** BPE merges digits based on corpus frequency, not mathematical significance. The number `710` may be one token while `711` is split as `71|1`. This creates unpredictable digit boundaries.
2. **Misaligned digit positions:** GPT-3.5/4 use left-to-right 3-digit chunking (cl100k_base). When the answer has more digits than operands (e.g., 999+1=1000), token boundaries shift, causing catastrophic failure. L2R tokenization accuracy drops to **8.25%** on length-mismatch problems.
3. **Obscured place values:** A token like `123` does not expose the individual digit positions to attention heads. The model must implicitly decompose multi-digit tokens.
4. **Non-uniform vocabulary:** Different numbers of the same digit-count get different tokenizations, preventing consistent learned algorithms.

### Tokenization Approaches by Model Family
| Model | Number Tokenization | Notes |
|-------|-------------------|-------|
| PaLM, LLaMA, Mistral | Single-digit tokens | Systematic, predictable |
| GPT-3.5, GPT-4 | L2R 3-digit chunks | Efficient but problematic for arithmetic |
| Claude, Gopher | Pure BPE | Idiosyncratic |

### The Comma Trick
Inserting commas every 3 digits from the right (e.g., `8,302,080`) forces right-to-left tokenization alignment. This improved GPT-3.5 8-shot accuracy from **68.5% to 97.8%** -- a dramatic gain from formatting alone.

---

## 4. Fixed-Width Zero-Padding

**Core idea:** Pad all numbers to the same width with leading zeros. E.g., `0123 + 0456 = 0579`.

### Findings
- Guarantees strict alignment of corresponding digit positions across operands and sum.
- Lee et al. initially used zero-padding but later found that wrapping with `$` delimiters was equally effective with less overhead.
- **ABA-fixed** format guarantees strict alignment; **ABA-var** preserves weaker alignment but works with standard inference (no padding needed at test time).
- Newer methods (Abacus Embeddings, position coupling) have largely superseded explicit zero-padding by encoding positional information in the embedding layer.

---

## 5. Index Hints / Position Markers

**Core idea:** Annotate each digit with its positional significance. E.g., `a4b2 + a3b9 = a8b1` for `42 + 39 = 81`.

### Key Papers
- **Nogueira et al. (2021)** -- Position tokens like `3 10e1 2` enabled addition up to 60 digits (vs. failure at 5 digits without them).
- **Zhou et al. (2024)** -- Index hinting generalizes to ~50-digit additions but fails at 70 digits.
- **McLeish et al. (2024)** "Transformers Can Do Arithmetic with the Right Embeddings" -- Abacus Embeddings assign same positional encoding to digits of equal significance. Achieves 99% accuracy on 100-digit addition, 6x length generalization (train on 20 digits, test on 120).
- **Cho et al. (2024)** "Position Coupling" (NeurIPS 2024) -- Assigns identical position IDs to tokens of same significance across operands and result. Achieves >95% exact-match accuracy on 200-digit additions trained on 30-digit additions (6.67x extrapolation).

### Index Hints vs. Position Coupling
| Method | Max Generalization | Overhead | Training Cost |
|--------|-------------------|----------|---------------|
| Index hints (Zhou 2024) | ~50 digits | 2x sequence length | 4x time/memory |
| Abacus Embeddings (McLeish 2024) | 120 digits (6x) | None (in embedding) | Standard |
| Position Coupling (Cho 2024) | 200 digits (6.67x) | None (in position IDs) | Standard |

### Theoretical Result
**Theorem (Cho et al.):** A single-layer, two-head Transformer with coupled positions can solve addition with operands up to 2^floor((d-17)/2) - 2 digits, where d is embedding dimension >= 21.

---

## 6. Scratchpad / Chain-of-Thought Tokenization

**Core idea:** Include intermediate computation steps in the training target. The model learns to emit step-by-step work before the final answer.

### Key Papers
- **Nye et al. (2021)** "Show Your Work: Scratchpads for Intermediate Computation" -- seminal paper showing scratchpads dramatically improve multi-step computation in language models.
- **Lee et al. (2024)** -- tested simplified and detailed scratchpad formats.

### Scratchpad Formats (Lee et al. results for 3-digit addition)
| Format | Samples to 100% | Notes |
|--------|-----------------|-------|
| Plain (MSB-first) | Never (plateaus 85%) | Baseline |
| Reversed (LSB-first) | ~2,500 | Simple formatting change |
| Simplified scratchpad | ~2,000 | Digit-sum + carry per step |
| Detailed scratchpad | ~1,000 | Natural language steps |

### Why Scratchpads Help
- Decompose a multi-step problem into single-step subproblems, each within the model's capacity.
- Provide explicit carry information in the context, eliminating the need to implicitly track carries across positions.
- Each generation step requires only local information (current digit pair + carry from previous step).

### Limitations
- Scratchpads increase sequence length substantially (3--10x depending on format).
- Training data must include correct intermediate steps.
- For addition/subtraction, reversed digit order alone may be sufficient (LEFT paper shows Little-Endian without step-by-step outperforms step-by-step alone).
- For multiplication, scratchpads remain essential -- LEFT multiplication without step-by-step fails entirely.
- Scratchpad design matters: poorly designed scratchpads can increase RASP-L program complexity and *hinder* generalization.

---

## 7. Complementary Architectural Techniques

While not tokenization per se, these architectural choices interact strongly with tokenization:

### Abacus Embeddings + Looped Transformers
- Combining Abacus Embeddings with input injection and looped (weight-sharing) transformers achieved 99.1% OOD accuracy (87% error reduction over standard transformers).
- Looped transformers reduce error by ~50% vs. standard stacked architectures in OOD settings.

### Recurrence and Weight Sharing
- Looped transformer layers effectively give the model more "thinking steps" without more parameters.
- Particularly helpful for carry propagation across many positions.

---

## Summary: Tokenization Strategy Rankings for Arithmetic

For a small LLM doing addition/subtraction/multiplication of integers:

### Tier 1 (Essential)
1. **Digit-level tokenization** -- each digit is its own token
2. **Reversed (LSB-first) output** -- aligns generation order with computation order

### Tier 2 (Strong Boosters)
3. **Position-aware embeddings** (Abacus or Position Coupling) -- for length generalization
4. **Scratchpad/CoT for multiplication** -- multiplication requires explicit intermediate steps

### Tier 3 (Helpful but Superseded)
5. **Zero-padding** -- useful but largely replaced by position-aware embeddings
6. **Index hints** -- effective but doubled sequence length; superseded by position coupling

### Anti-Pattern
7. **BPE/subword tokenization** -- actively harmful for arithmetic due to inconsistent digit boundaries

---

## Source URLs
- Lee et al. (2024): https://arxiv.org/abs/2307.03381
- McLeish et al. (2024): https://arxiv.org/abs/2405.17399
- Cho et al. (2024): https://arxiv.org/abs/2405.20671
- "Reverse That Number" / LEFT: https://arxiv.org/abs/2403.05845
- "Tokenization Counts": https://arxiv.org/abs/2402.14903
- Stolfo et al. (2024): https://arxiv.org/abs/2402.02619
- RevOrder (Shen 2025): https://arxiv.org/abs/2402.03822
- Nogueira et al. (2021): https://mathai-iclr.github.io/papers/papers/MATHAI_11_paper.pdf
- Nye et al. (2021): https://arxiv.org/abs/2112.00114
- Zhou et al. (2024) Length Generalization: https://arxiv.org/abs/2306.15400
