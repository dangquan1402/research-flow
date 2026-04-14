---
title: "Architecture Sizing for Arithmetic Transformers"
created: 2026-04-13
type: web-search
ingested_by: "hypothesis/GH-3-architecture-sizing"
---

# Architecture Sizing for Arithmetic Transformers

## Research Summary

This document aggregates findings from web research on optimal transformer architecture sizing for learning simple arithmetic (addition, subtraction, multiplication of integers), targeting <100M parameters trainable on consumer hardware.

## Key Papers and Sources

### 1. Teaching Arithmetic to Small Transformers (Lee & Sreenivasan, 2023)
- **URL:** https://arxiv.org/abs/2307.03381
- **Architecture:** NanoGPT — decoder-only, 6 layers, 6 heads, embedding dim 384, ~10.6M parameters
- **Tokenization:** Character-level with absolute position encoding
- **Key finding:** Data formatting matters more than architecture. Reversing output to LSB-first enables a phase transition at ~2500 training samples where addition is learned perfectly. Plain MSB-first format plateaus at ~85% with 10K samples.
- **Techniques:** Reverse formatting, simplified/detailed scratchpad (chain-of-thought), structured sampling by digit frequency and carry operations
- **Scaling:** Extended experiments to GPT-2 (124M) and GPT-3

### 2. Understanding Addition and Subtraction in Transformers (Nikankin et al., 2024)
- **URL:** https://arxiv.org/abs/2402.02619
- **Architecture:** 2 layers, 3 attention heads — minimal configuration for >99.999% accuracy on multi-digit addition
- **Key finding:** Transformers partition arithmetic into parallel digit-specific operations. Three heads operate with time-offset of 1 token, each doing independent calculations on relevant digit pairs. Results are combined by the MLP layer.
- **Carry mechanism:** Models develop 3 distinct clusters in PCA corresponding to carry states (definite carry, no carry, uncertain). Specialized nodes handle: base addition (SA), single-digit carry detection (ST), cascading carry resolution (SV), and answer finalization.

### 3. Transformers Can Do Arithmetic with the Right Embeddings (McLeish et al., NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.17399
- **Architecture:** Tested standard (16 layers), input injection, and looped transformers. Parameters 12M-122M.
- **Abacus Embeddings:** Learned embeddings encoding digit positions relative to number starts, with randomized offsets during training. Aligns digits of equal significance across numbers.
- **Key result:** Training on 20-digit numbers achieved 6x length generalization to 100+ digits. Combining Abacus Embeddings + input injection + looped transformers: 92.9% -> 99.1% on OOD tasks.
- **Looped transformers:** Weight-tied recurrent layers (1x16 through 16x1 configs), explicitly decoupling computational depth from parameter count.

### 4. Position Coupling (Cho et al., NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.20671
- **Architecture:** 1-layer 4-head and 6-layer 8-head models tested
- **Method:** Assigns same position IDs to digits of equal significance across operands/results (rather than unique sequential IDs)
- **Result:** Models trained on 1-30 digit additions generalize to 200-digit additions (6.67x extrapolation)
- **Theoretical:** 1-layer transformer with coupled positions can solve addition for exponentially many digits. Without positional info, 1-layer cannot.
- **Multiplication:** 2-layer models with 10 attention heads needed for Nx2 multiplication. Minimum embedding dim >= 46.

### 5. Dissecting Multiplication in Transformers (2024)
- **URL:** https://arxiv.org/abs/2407.15360
- **Architecture:** "Tiny transformer" — 3 attention heads sufficient for multiplication
- **Key techniques:** (1) Reversed answer format for carry calculation, (2) Increased depth for intermediate steps, (3) Progressive training (increase simple sample proportion)
- **Result:** >99.9% accuracy on 5-digit multiplication, outperforming GPT-4
- **Insight:** Transformers decompose multiplication into parallel subtasks with different heads managing base multiplication, carry calculation, and combined operations

### 6. Why Can't Transformers Learn Multiplication? (2025)
- **URL:** https://arxiv.org/abs/2510.00184
- **Key finding:** Even 12-layer models fail to learn right long-range dependencies for multiplication. Middle digits never receive correct gradients. Auxiliary loss terms for intermediate sums help: 2-layer model achieves 99% on 4x4 multiplication with auxiliary losses.

### 7. The Depth Delusion (2026)
- **URL:** https://arxiv.org/abs/2601.20994
- **Key finding:** Beyond critical depth D_crit ~ W^0.44, adding layers increases loss despite adding parameters. Optimal depth scales as D* ~ C^0.12, optimal width as W* ~ C^0.34. Width should grow 2.8x faster than depth.
- **Mechanism:** Gradient starvation — exponential decay of gradient signal through deep layers.
- **Validated:** Across 30 architectures, 17M-7B params (R^2 = 0.922).

### 8. Depth-Width Tradeoffs in Algorithmic Reasoning (2025)
- **URL:** https://arxiv.org/abs/2503.01805
- **Key finding:** With linear width, constant depth suffices for many graph-based algorithmic problems. Wider models match accuracy of deep models with faster train/inference due to parallelization.

### 9. Grokking on Modular Arithmetic
- **URL:** https://ar5iv.labs.arxiv.org/html/2201.02177 (Power et al., 2022)
- **Architecture:** 2 layers, 4 heads, 128 embedding dim — standard for grokking studies
- **Key finding:** Grokking (delayed generalization after memorization) occurs with very small transformers on modular arithmetic. Phenomenon is contingent on model size, weight decay, and data size being "just right."
- **MLX implementation:** 2 layers, 1 head, 128 dim achieved grokking in <150 epochs on modular division

### 10. Impact of Positional Encoding on Length Generalization (Kazemnejad et al., NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.19466
- **Key finding:** Standard PE methods (ALiBi, RoPE, APE) are not well suited for length generalization on downstream tasks. NoPE (no positional encoding) outperforms explicit PE methods while requiring no additional computation.
- **Tested:** APE, T5 Relative PE, ALiBi, RoPE, and NoPE across multiple tasks

## Architecture Comparison Table

| Paper | Task | Layers | Heads | Embed Dim | Params | Accuracy |
|-------|------|--------|-------|-----------|--------|----------|
| Lee 2023 | Addition | 6 | 6 | 384 | 10.6M | ~100% (reversed) |
| Nikankin 2024 | Addition | 2 | 3 | — | ~10M | 99.999% |
| McLeish 2024 | Addition | 16 (looped) | — | — | 12-122M | 99.1% OOD |
| Cho 2024 | Addition | 1-6 | 4-8 | — | — | >95% @ 200 digits |
| Cho 2024 | Multiplication | 2 | 10 | >=46 | — | good |
| 2024 | 5-digit Mult | few | 3 | — | tiny | 99.9% |
| Grokking | Modular Arith | 2 | 4 | 128 | small | grokking |
| nanoGPT (Karpathy) | Shakespeare | 4 | 4 | 128 | ~85K | educational |

## Key Themes

1. **Data formatting > architecture**: Reversed output, chain-of-thought, structured sampling matter more than raw model size
2. **2 layers suffice for addition**: Consistent finding across papers
3. **3 heads minimum for carry**: Heads specialize in base computation, carry detection, carry cascading
4. **Positional encoding is critical for generalization**: Standard methods fail; task-specific (Position Coupling, Abacus) excel
5. **Multiplication is fundamentally harder**: Requires more depth, auxiliary losses, or progressive training
6. **Width > depth for small models**: Gradient starvation makes deep narrow models worse
7. **Looped/recurrent transformers** offer parameter efficiency without sacrificing computational depth
8. **Pre-norm is standard** for small model training stability

## Normalization

- Pre-norm (Pre-LN) is the default for modern transformers including small ones
- Eliminates need for learning rate warmup
- Post-norm is unstable for small models, requires careful hyperparameter tuning
- All major LLMs (GPT-2, GPT-3, LLaMA, Falcon, Mistral) use Pre-norm
- Pre-norm outperforms Post-norm especially in low-resource settings

## FFN and Parameter Distribution

- Standard FFN expansion ratio: 4x embedding dim
- SwiGLU-based models use ~2.7x (gating effectively doubles transformation)
- FFN contains ~2/3 of each layer's parameters
- Head dimension typically 64-128 for good balance
- Embedding layers can dominate parameter count for large vocabularies (less relevant for character-level arithmetic)

## Positional Encoding Summary

| Method | Length Gen | Params | Training Speed | Adoption |
|--------|-----------|--------|---------------|----------|
| Learned/APE | None | O(L*d) | Baseline | GPT-2 |
| Sinusoidal | Limited | 0 | Fast | Original Transformer |
| RoPE | Moderate | 0 | Fast | LLaMA, Mistral |
| ALiBi | Better than RoPE | 0 | Fastest | BLOOM |
| NoPE | Surprisingly good | 0 | Fast | Research |
| Position Coupling | Excellent (6.67x) | minimal | Fast | Research |
| Abacus Embeddings | Excellent (6x) | small | Fast | Research |

## Training Compute

- NanoGPT (10.6M params): trains under 1 hour on single GPU
- nanoGPT minimal config (4L/4H/128d, ~85K params): ~3 minutes on CPU
- GPT-2 124M reproduction: requires 8x A100 40GB
- For arithmetic specifically: 2L/3H models at <10M params train very quickly on consumer hardware
- Karpathy's nanochat: "mini ChatGPT" trained in 4 hours for ~$100 on consumer GPU
