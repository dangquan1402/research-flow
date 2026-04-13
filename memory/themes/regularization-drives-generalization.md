---
title: "Theme: Regularization Is the Engine of Arithmetic Generalization"
created: 2026-04-13
updated: 2026-04-13
source: analysis
confidence: high
verification: analysis
tags: [training, regularization, grokking, weight-decay, optimizer]
synthesizes: [grokking-arithmetic, optimizer-schedule-arithmetic]
staleness_days: 0
---

## Pattern

Weight decay plays an outsized role in arithmetic transformer training compared to standard NLP. The grokking phenomenon — sudden generalization long after overfitting — is mechanistically understood as weight decay forcing the model from a high-norm memorization solution to a low-norm algorithmic (Fourier) circuit. The optimal weight decay for arithmetic (0.1-1.0) is 10-100x higher than typical NLP defaults (0.01). This, combined with AdamW's decoupled weight decay and an appropriate LR schedule, forms the core training recipe.

The training dynamics have three phases:
1. **Memorization** (fast): model stores input-output pairs with large weights
2. **Circuit formation** (slow): algorithmic solution develops alongside memorization
3. **Cleanup** (sudden): weight decay prunes memorization weights, revealing the circuit — this is the "grokking" moment

## Supporting Findings

- **grokking-arithmetic**: Grokking is real and mechanistically understood (Nanda et al. 2023). The model learns a discrete Fourier transform circuit for modular arithmetic. Weight decay is the single most important lever — it more than halves samples needed for generalization. Three-phase mechanism: memorization → circuit formation → cleanup.
- **optimizer-schedule-arithmetic**: AdamW with lr=1e-3, beta2=0.99, weight_decay=0.5, batch 256, cosine schedule with 100-step warmup is the established recipe. Beta2=0.99 (vs default 0.999) helps with the sharp loss landscape. Pre-norm eliminates the warmup requirement but keeping it doesn't hurt.

## Contradictions

- **Grokking requires long training, but good format reduces training time**: With reversed output, the model may generalize during normal training (no grokking needed). Grokking is most relevant when format is suboptimal. **Resolution**: Use reversed output + high weight decay. If the model generalizes quickly, great. If it plateaus on validation, the high weight decay will eventually trigger grokking. The two are complementary, not contradictory.
- **Grokking is studied on modular arithmetic, not general integers**: The Fourier circuit mechanism is specific to finite groups. Unbounded integer arithmetic may use different generalization mechanisms. **Resolution**: This is an open question (GH open-questions/grokking-on-integer-vs-modular-arithmetic.md). Set high weight decay regardless — even without grokking, it regularizes against memorization.

## Counter-arguments

- **High weight decay may hurt final performance**: Too much regularization can prevent the model from learning complex patterns. **Disproof condition**: If accuracy plateaus below target with wd=0.5, try reducing to 0.1.
- **Grokking may be an artifact of small scale**: The phenomenon is best documented on tiny models with tiny datasets. It may not be relevant at our scale (10M params, millions of examples). **Disproof condition**: If the model generalizes smoothly during normal training without a grokking phase, this theme is less actionable.

## So What?

**For our training recipe**:
1. Use AdamW with weight_decay=0.5 (tune in [0.1, 1.0])
2. lr=1e-3, beta2=0.99, batch_size=256
3. Cosine schedule with 100-step linear warmup
4. Monitor validation loss well beyond training convergence — don't early-stop prematurely
5. Track weight norm / gradient norm ratio as a grokking diagnostic
6. For modular arithmetic subtasks, expect and plan for grokking dynamics
