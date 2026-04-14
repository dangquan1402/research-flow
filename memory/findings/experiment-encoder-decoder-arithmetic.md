---
title: "Experiment: Encoder-Decoder Architecture for Arithmetic"
created: 2026-04-14
updated: 2026-04-14
source: experiment
confidence: high
tags: [architecture, encoder-decoder, addition, multiplication, scratchpad]
---

## Encoder-Decoder vs Decoder-Only for Arithmetic

### Architecture
- **Encoder**: 1 layer bidirectional attention over equation prefix (up to `=`)
- **Decoder**: 1 layer causal self-attention + cross-attention to encoder + FFN
- Implementation uses "prefix LM" masking approach on full sequence for efficiency
- Parameters: 4.15M (vs 3.57M for 2L decoder-only — extra ~590K from cross-attention)

### Results: 5-Digit Addition

| Metric | Encoder-Decoder (1L+1L) | Decoder-Only (2L/384D) |
|---|---|---|
| Params | 4.15M | 3.57M |
| First 100% | Epoch 15 | Epoch 20 |
| Final accuracy | 99.65% (epoch 50) | 99.9% (epoch 20) |
| Stability | Dip to 47.5% at epoch 25 | Stable |

**Finding**: Encoder-decoder reaches 100% faster on addition (epoch 15 vs 20) but shows instability — a transient accuracy collapse at epoch 25 before recovering. The bidirectional encoder helps the model understand the equation structure earlier.

### Results: 3-Digit Multiplication + Scratchpad

| Metric | Encoder-Decoder (1L+1L) | Decoder-Only (2L/384D) |
|---|---|---|
| Params | 4.16M | 3.57M |
| Final accuracy | 87.8% (epoch 80) | 100% (epoch 35) |
| 5-digit accuracy | 76% | 100% |
| 6-digit accuracy | 58% | 100% |
| Convergence | Plateaued ~88% | Full convergence |

**Finding**: Encoder-decoder significantly underperforms on multiplication with scratchpad. The model plateaus at ~88% and cannot solve longer output sequences (5-6 digit results). This suggests the cross-attention bottleneck limits the decoder's ability to perform multi-step reasoning in scratchpad format.

### Key Insights

1. **Addition benefits from bidirectional encoding**: The encoder's ability to see the full equation bidirectionally helps with addition, where digit alignment is the core challenge.

2. **Scratchpad hurts encoder-decoder**: The scratchpad format requires extended autoregressive generation (step-by-step intermediate computations). The cross-attention bottleneck means the decoder must compress all equation information through attention to encoder states, rather than having direct access in its context (as decoder-only does).

3. **Instability in training**: The encoder-decoder shows more training instability (epoch 25 addition collapse) compared to decoder-only. The interaction between encoder and decoder training dynamics may create competing gradients.

4. **Parameter efficiency**: Despite having ~16% more parameters, the encoder-decoder underperforms decoder-only on multiplication. The additional cross-attention parameters don't compensate for the architectural constraint.

### What Would Disprove This?
- A deeper encoder (2+ layers) might provide richer representations that improve multiplication
- Different cross-attention patterns (e.g., multi-query attention) might reduce the bottleneck
- Separate learning rates for encoder and decoder might reduce instability
