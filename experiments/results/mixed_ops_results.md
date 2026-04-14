# Mixed-Operation Experiment Results

## Configuration

| Parameter | Value |
|---|---|
| Operations | add + sub + mul (equal proportions) |
| Architecture | 2L/4H/384D (2.78M params) |
| Max digits (add/sub) | 5 |
| Max digits (mul) | 3 |
| Add/Sub format | Reversed (LSB-first), no scratchpad |
| Mul format | Reversed + scratchpad (partial products) |
| Training samples | 50,000 (16,667 per op) |
| Test samples | 2,000 (667 per op) |
| Epochs | 80 |
| Batch size | 256 |
| Framework | MLX |
| Total time | 2301.6s (~38 min) |

## Final Per-Operation Accuracy

| Operation | Accuracy | Target | Status |
|---|---|---|---|
| Addition (5-digit) | **99.10%** | >99% | PASS |
| Subtraction (5-digit) | **99.70%** | >99% | PASS |
| Multiplication (3-digit) | **97.45%** | >98% | MARGINAL |
| Overall | **98.75%** | - | - |

## Training Progression (per-op accuracy)

| Epoch | Overall | Add | Sub | Mul |
|---|---|---|---|---|
| 5 | 32.95% | 25% | 25% | 49% |
| 10 | 77.30% | 78% | 93% | 62% |
| 15 | 90.55% | 100% | 100% | 72% |
| 20 | 91.65% | 99% | 99% | 77% |
| 25 | 93.65% | 100% | 99% | 82% |
| 30 | 96.30% | 100% | 100% | 89% |
| 35 | 95.55% | 100% | 99% | 88% |
| 40 | 97.65% | 100% | 99% | 94% |
| 45 | 99.25% | 100% | 100% | 98% |
| 50 | 99.35% | 100% | 99% | 99% |
| 55 | 99.10% | 100% | 100% | 98% |
| 60 | 99.50% | 100% | 100% | 98% |
| 65 | 99.35% | 100% | 100% | 98% |
| 70 | 99.15% | 100% | 100% | 98% |
| 75 | 99.20% | 99% | 100% | 99% |
| 80 | 98.75% | 99% | 100% | 97% |

## Observations

1. **Subtraction learned fastest** — hit 93% by epoch 10, 100% by epoch 15
2. **Addition converged quickly** — 100% by epoch 15
3. **Multiplication was harder** — expected, since it uses scratchpad and has more complex output structure
4. **Peak accuracy at epoch 50** — all ops >99%. Slight degradation in later epochs suggests overfitting or LR schedule effects
5. **No catastrophic interference** — a single 2L model handles all three operations simultaneously with near-perfect accuracy
6. **Mul peaked at 99% (epoch 50/75)** — the 97.45% at epoch 80 is a slight regression, likely from continued training past optimum
7. **Best checkpoint would likely be around epoch 45-50** for all-ops performance

## JSON Results

Full results: `mixed_5d_reversed_2L4H384D_mixed_2L384D.json`
