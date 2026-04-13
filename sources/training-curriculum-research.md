---
title: "Training Curriculum and Data Strategy for Arithmetic LLMs"
created: 2026-04-13
type: web-search
ingested_by: "hypothesis/GH-4-training-curriculum"
---

# Training Curriculum and Data Strategy for Arithmetic LLMs

## 1. Data Format

The format in which arithmetic problems are presented to the model has a decisive impact on accuracy and sample efficiency. Four main approaches have been studied:

1. **Plain format**: `123+456=579` -- standard left-to-right. Never reaches 100% accuracy on multi-digit addition in isolation.
2. **Reverse format (Little-Endian)**: Output digits are reversed so the least significant digit is predicted first. This aligns the autoregressive generation order with the natural order of carry propagation. RevOrder achieved 100% accuracy on addition, subtraction, and multiplication, outperforming plain format while using only 5.2% of the training tokens. ([Shen et al., "RevOrder", 2024](https://arxiv.org/html/2402.03822); [Gao et al., "Reverse That Number!", 2024](https://arxiv.org/html/2403.05845v1))
3. **Simplified Scratchpad**: Records digit-wise sum and carry-ons as intermediate tokens (e.g., `3+5=c0s8`). Improves accuracy and convergence.
4. **Detailed Scratchpad / Chain-of-Thought**: Full intermediate computation steps. Best accuracy and sample efficiency, but increases sequence length significantly.

**Key insight**: Plain format forces the model to predict the most significant digit first, before it has information about carries. Reverse and scratchpad formats resolve this by either reordering output or providing intermediate state.

Sources:
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://arxiv.org/abs/2307.03381)
- [Shen et al., "RevOrder: A Novel Method for Enhanced Arithmetic in Language Models", 2024](https://arxiv.org/html/2402.03822)
- [Gao et al., "Reverse That Number!", 2024](https://arxiv.org/html/2403.05845v1)

## 2. Curriculum Learning

Curriculum learning (training on easy examples first, then harder ones) has mixed evidence for arithmetic:

- **Progressive digit curriculum**: Train on 1-digit addition until ~96% accuracy, then switch to 2-digit (accuracy initially drops to ~5%), continue until 90%+, then 3-digit, etc. This works but is not always strictly necessary.
- **Position curriculum**: Broadening the range of absolute character positions seen during training. Combined with template diversity and expression boundary markers, this significantly improves robustness to input-format variation. ([Chen et al., 2025](https://arxiv.org/abs/2601.04283))
- **Counterintuitive dynamics**: Recent work shows transformers can exhibit "shattered compositionality" where curriculum ordering has unexpected effects -- models sometimes learn component skills in non-intuitive orders. ([Liu et al., "Shattered Compositionality", 2025](https://arxiv.org/html/2601.22510))

The consensus is that curriculum learning helps with sample efficiency but is not strictly required if the data format and sampling strategy are good enough.

Sources:
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://arxiv.org/abs/2307.03381)
- [Chen et al., "Mitigating Position-Shift Failures", 2025](https://arxiv.org/abs/2601.04283)
- [Liu et al., "Shattered Compositionality", 2025](https://arxiv.org/html/2601.22510)

## 3. Number Distribution and Balanced Sampling

When sampling operands uniformly at random, the training data becomes heavily skewed:
- Most numbers have the maximum number of digits (e.g., 90% of numbers between 0-999 have 3 digits)
- Long carry chains are extremely rare in random samples
- Certain carry patterns are severely underrepresented

**Balanced carry sampling** addresses this by: (1) uniformly sampling the carry chain length, then (2) sampling operands that produce that carry chain length. This ensures diversity in difficulty and is critical for length generalization. Models trained with balanced carry sampling show the first instance of strong length generalization on decimal addition for transformers trained from scratch.

**Balanced digit sampling** assigns higher weights to numbers with fewer digits to ensure uniform coverage across digit lengths.

Sources:
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://arxiv.org/abs/2307.03381)
- [McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings", 2024](https://arxiv.org/html/2405.17399v1)
- [Cho et al., "What Algorithms can Transformers Learn?", 2024](https://arxiv.org/pdf/2310.16028)

## 4. Length Generalization

Training on N-digit arithmetic and testing on N+k digits is a major challenge. Key techniques:

1. **Relative Position Embeddings (RPE)**: Enable generalization from 5-digit to 15-digit addition, but fail for multiplication.
2. **Train Set Priming**: Adding 10-50 long examples to training enables generalization to much longer inputs (e.g., 5x3 digit multiplication generalizes to 35x3). Priming sample size scales logarithmically with training set size.
3. **Attention Bias Calibration (ABC)**: A calibration stage where the model learns proper attention biases, achieving near-perfect length generalization on certain tasks.
4. **Position Coupling**: Assigns same position IDs to digits of the same significance across operands and result. A 1-layer Transformer with coupled positions can theoretically solve addition with exponentially many digits. Empirically achieves 95%+ accuracy on 200-digit addition when trained on up to 30-digit addition. ([Cho et al., 2024](https://arxiv.org/abs/2405.20671))
5. **Abacus Embeddings + FIRE**: Learned positional embeddings encoding positions within numerical spans, combined with FIRE position encoding.
6. **Randomized Positional Encodings**: Boost length generalization by exposing the model to varied position indices during training.
7. **Universal Transformers**: Shared-layer (recurrent) architectures are important for length generalization, especially on modular arithmetic tasks.

Sources:
- [Kazemnejad et al., "Length Generalization in Arithmetic Transformers", 2023](https://arxiv.org/abs/2306.15400)
- [Cho et al., "From Interpolation to Extrapolation", 2023](https://arxiv.org/abs/2310.11984)
- [Cho et al., "Position Coupling", 2024](https://arxiv.org/abs/2405.20671)
- [McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings", 2024](https://arxiv.org/html/2405.17399v1)

## 5. Optimizer Choice and Learning Rate Schedule

- **AdamW** is the standard optimizer for training transformers, including small arithmetic models. Weight decay in AdamW is critical -- it both regularizes the model and enables grokking (see section 8).
- **Learning rate**: The "Teaching Arithmetic to Small Transformers" paper uses lr=1e-3 with AdamW, beta2=0.99.
- **Schedule**: Warmup + cosine decay is standard. The Lee et al. paper uses 100 warmup iterations out of 10,000 total.
- **Component-specific LRs**: Recent research shows tuning learning rates separately for embedding, attention, and FFN layers improves performance. Rates tuned on small models transfer to larger ones.

Sources:
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://github.com/lee-ny/teaching_arithmetic)
- [HuggingFace Optimization Documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)

## 6. Batch Size

- Lee et al. use batch size 256 for their 10.6M parameter NanoGPT model.
- Larger batch sizes improve convergence in fewer steps but have diminishing returns beyond ~2048 tokens.
- Large-batch training can negatively impact generalization when the dataset is small.
- For grokking specifically, smaller batch sizes may actually help because they introduce more noise, which can aid in escaping memorization solutions.

Sources:
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://github.com/lee-ny/teaching_arithmetic)
- [Li et al., "Train Large, Then Compress", 2020](https://arxiv.org/abs/2002.11794)

## 7. Loss Function

- Standard **cross-entropy loss** on next-token prediction is the default and works well.
- **Per-digit weighting**: No widely published technique specifically for arithmetic, but the principle of weighted cross-entropy (upweighting harder or rarer digit positions) applies. The most significant digit and carry digits are hardest to predict; weighting these higher could help.
- The key insight from scratchpad research is that the loss function matters less than the **data format** -- if you restructure the data to include intermediate steps, standard cross-entropy suffices.
- For arithmetic specifically, the loss on the output digits is what matters, while the loss on input tokens (which the model memorizes trivially) adds noise. Some implementations mask the loss on input tokens.

## 8. Grokking

Grokking is the phenomenon where a model suddenly generalizes long after overfitting the training data. First observed by Power et al. (2022) on modular arithmetic.

**Mechanism** (Nanda et al., 2023): The model learns a discrete Fourier transform circuit -- projecting inputs to rotations on a circle and composing them. Training proceeds in three phases:
1. **Memorization**: Model memorizes training examples with large weights
2. **Circuit formation**: Structured algorithm gradually amplifies in the weights
3. **Cleanup**: Weight decay prunes the memorization components, revealing the generalizing circuit

**How to encourage grokking**:
- **Weight decay is essential**: It makes the memorization solution energetically expensive, pushing the model toward smaller-weight generalizing solutions. More weight decay = faster grokking (up to a point).
- **Small training set**: Grokking occurs most dramatically with small training fractions (e.g., 30-50% of all possible examples). Very small fractions require millions of steps.
- **Right model size**: Too small = can't learn the algorithm. Too large = memorization is too easy.
- **Long training**: Grokking can require 10-100x more steps than needed to overfit.

**Caution**: Grokking is a contingent phenomenon -- it disappears if hyperparameters aren't tuned correctly.

Sources:
- [Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", 2022](https://arxiv.org/abs/2201.02177)
- [Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability", 2023](https://arxiv.org/abs/2301.05217)
- [Google PAIR, "Do Machine Learning Models Memorize or Generalize?"](https://pair.withgoogle.com/explorables/grokking/)

## 9. Data Augmentation

- **Commutativity**: For addition and multiplication, including both `a+b=c` and `b+a=c` doubles the effective dataset and encodes an inductive bias the model otherwise must learn.
- **Identity operations**: Including `a+0=a` and `a*1=a` helps anchor number representations.
- **MAMUT (Mathematical Augmentation through Universal Transformations)**: Uses additive and multiplicative commutativity to derive equivalent formulas. Models trained on MAMUT-enhanced data outperform baselines.
- **Operand padding**: Padding shorter operands with leading zeros (e.g., `007+123`) helps models learn positional alignment.

Sources:
- [MAMUT, 2025](https://www.arxiv.org/pdf/2502.20855)

## 10. Scratchpad / Chain-of-Thought

Scratchpad training dramatically improves multi-step computation by allowing models to emit and condition on intermediate tokens.

**Approaches**:
- **Simplified scratchpad**: `123+456 -> c1:0,s1:9,c2:0,s2:7,c3:0,s3:5 -> 579`
- **Detailed scratchpad**: Full step-by-step showing each column addition with carry
- **Inductive scratchpad**: For overcoming the "globality barrier" where standard scratchpad fails

**Theoretical power**: Chain of thought with linear steps keeps transformer decoders within context-sensitive languages; polynomial steps enable recognition of all polynomial-time problems.

**Trade-offs**:
- Significantly increases sequence length (3-10x for detailed scratchpad)
- Requires more compute per example
- But dramatically improves accuracy, sample efficiency, and convergence speed
- Works even without pretraining (from random initialization)

Sources:
- [Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with Language Models", 2022](https://openreview.net/forum?id=iedYJm92o0a)
- [Lee et al., "Teaching Arithmetic to Small Transformers", 2023](https://arxiv.org/abs/2307.03381)
- [Feng et al., "The Expressive Power of Transformers with Chain of Thought", 2023](https://arxiv.org/html/2310.07923)
- [Bartan et al., "How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad", 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/3107e4bdb658c79053d7ef59cbc804dd-Paper-Conference.pdf)

## Summary: Recommended Training Recipe

Based on the literature, a strong baseline recipe for training a small arithmetic LLM:

1. **Architecture**: Decoder-only transformer, ~10M params, 6 layers, 6 heads, embedding dim 384
2. **Data format**: Reverse (Little-Endian) output OR simplified scratchpad. Avoid plain format.
3. **Sampling**: Balanced carry sampling + balanced digit sampling
4. **Position encoding**: Position coupling or Abacus embeddings for length generalization; at minimum use relative position embeddings
5. **Optimizer**: AdamW, lr=1e-3, beta2=0.99, weight decay=0.1-1.0
6. **Schedule**: Linear warmup (100 steps) + cosine decay
7. **Batch size**: 256
8. **Training**: 10K-100K iterations depending on task complexity
9. **Curriculum**: Optional progressive digit curriculum; more important to get data format and sampling right
10. **Augmentation**: Include commutative variants, identity operations, zero-padded operands
