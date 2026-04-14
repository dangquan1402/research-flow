---
title: "Finding: Aligned Partial Products Is the Optimal Scratchpad Format for Multiplication"
created: 2026-04-13
updated: 2026-04-13
source: scratchpad-format-research
confidence: high
verification: analysis
tags: [multiplication, scratchpad, tokenization, partial-products, chain-of-thought]
related: [scratchpad-chain-of-thought, scratchpad-chain-of-thought-arithmetic, reversed-digit-order]
staleness_days: 0
---

## Insight

The optimal scratchpad format for training small transformers on multiplication is **aligned partial products with reversed digit order**: each digit of the multiplier × the full multiplicand, shifted by position (zeros prepended in reversed form), followed by the final reversed sum. This decomposes multiplication into sub-problems the model already knows: single-digit × multi-digit (easy), positional alignment (mechanical zero-prepend), and multi-operand addition (already mastered via reversed output).

## Evidence

- **Lee et al. (ICLR 2024)**: Four format levels tested (Plain < Reverse < Simplified Scratchpad < Detailed Scratchpad). Detailed scratchpad with intermediate steps achieves best accuracy and sample efficiency.
- **ICoT paper (Deng et al. 2024)**: Uses reversed partial products with `+` separator: `8 4 + 0 6 3` → `8 0 4` for 12×34. Achieves 100% accuracy on 4×4 digit multiplication with ICoT training. Standard fine-tuning without intermediate steps achieves <1%.
- **"Why Can't Transformers Learn Multiplication" (2024)**: Full format with running sums in parentheses and special delimiters. LSB-first ordering essential. Running sums help but add significant token overhead.
- **Merrill & Sabharwal (2023)**: Linear CoT steps (proportional to input length) extend transformers to all regular languages. Multiplication partial products are O(N) steps for N-digit multiplier.
- **LEFT paper (2024)**: Confirms both little-endian AND step-by-step are essential for multiplication — without either, learning fails entirely.

## Recommended Format

**Aligned reversed partial products with `|` separator:**

Example: `12 × 34 = 408`
```
1 2 * 3 4 = 8 4 | 0 6 3 | 8 0 4 <EOS>
```

Breakdown:
- `8 4` = 4 × 12 = 48, reversed → "84" (no shift for position 0)
- `0 6 3` = 3 × 12 = 36 → 360 (shifted by 1), reversed → "063"
- `8 0 4` = final sum 408, reversed → "804"

Example: `123 × 456 = 56088`
```
1 2 3 * 4 5 6 = 8 3 7 | 0 5 1 6 | 0 0 2 9 4 | 8 8 0 6 5 <EOS>
```

### Why This Format

1. **Each partial product is single-digit × multi-digit** — the simplest possible multiplication sub-task
2. **Shift alignment via zero-prepend in reversed order** — mechanical and position-invariant
3. **Final sum is multi-operand addition** — the model already masters reversed addition
4. **Minimal token overhead** — no running sums, no redundant delimiters. Only M partial products + 1 final answer for an N×M multiplication
5. **Compatible with reversed output** — all digits are LSB-first, consistent with existing format

### Token Overhead

For N×M digit multiplication:
- Without scratchpad: ~N+M output digits
- With scratchpad: M partial products (each N+1+position digits) + M separators + final answer
- For 3×3: ~18 scratchpad tokens + 6 answer tokens = ~24 total (vs ~6 without) = **4× overhead**
- For 5×5: ~65 scratchpad tokens + 10 answer = ~75 total (vs ~10 without) = **7.5× overhead**

### Alternatives Considered and Rejected

1. **Running sums after each partial product**: Adds tokens without proportional accuracy gain. The model can learn to accumulate implicitly or in the final step.
2. **Digit-by-digit decomposition**: O(N×M) steps instead of O(M). Too verbose for 3+ digit multiplication.
3. **No shift zeros (compact partial products)**: Requires the model to learn implicit positional shifting — harder than explicit zero-padding.
4. **Lattice method**: Not naturally sequential; hard to linearize for autoregressive generation.

## Counter-arguments

- **4-7× sequence length overhead** increases training compute quadratically via attention.
- **Error propagation**: Wrong intermediate step → wrong final answer. But this is inherent to any scratchpad.
- **May not be needed for very small digit counts** (1×1, 2×1) where the model can compute directly.
- **ICoT approach** (progressively removing intermediate tokens) could reduce inference cost, but adds training complexity.

## Implications

- Add `|` separator token to vocabulary (VOCAB_SIZE: 17 → 18)
- Generate training data with partial products for multiplication when `--scratchpad` is enabled
- Keep non-scratchpad mode as default for backward compatibility
- Increase max_seq_len cap from 64 to 128 for scratchpad mode
- Expected accuracy improvement: 94.9% → >99% (4L/256D), 85.15% → potentially >95% (2L/384D)
