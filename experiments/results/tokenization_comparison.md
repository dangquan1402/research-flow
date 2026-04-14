# Tokenization Comparison: Reversed vs Plain Output

## Experiment Design

Same model (4L/4H/256D, 3.16M params), same training data (50K samples, 5-digit addition, balanced carry), same hyperparameters. Only difference: output digit order.

| Condition | Final Accuracy | Peak Accuracy | Epoch at 95% |
|---|---|---|---|
| **Reversed (LSB-first)** | **99.90%** | **100.00%** (ep 45) | Epoch 15 |
| Plain (MSB-first) | 97.85% | 97.85% (ep 50) | Epoch 30 |

## Detailed Comparison

### Accuracy by digit count at epoch 50

| Digits | Reversed | Plain | Delta |
|---|---|---|---|
| 1 | 100% | 100% | 0% |
| 2 | 100% | 99% | +1% |
| 3 | 100% | 98% | +2% |
| 4 | 100% | 98% | +2% |
| 5 | 100% | 97% | +3% |
| 6 | 100% | 98% | +2% |

### Learning speed comparison

| Epoch | Reversed | Plain |
|---|---|---|
| 5 | 49.45% | 29.55% |
| 10 | 86.95% | 72.90% |
| 15 | 95.50% | 83.40% |
| 20 | 93.15% | 90.45% |
| 25 | 97.90% | 89.35% |
| 30 | 97.60% | 95.95% |
| 35 | 99.00% | 95.55% |
| 40 | 93.85% | 96.95% |
| 45 | 100.00% | 97.45% |
| 50 | 99.90% | 97.85% |

## Key Findings

1. **Reversed output provides +2.05% final accuracy** (99.90% vs 97.85%).
2. **Reversed learns ~2x faster** — reaches 95% at epoch 15 vs epoch 30 for plain.
3. **Reversed achieves perfect accuracy** (100% at epoch 45); plain never reaches it.
4. **The gap is largest for higher digit counts**, consistent with carry propagation theory.
5. **Plain format still achieves strong results** (97.85%) with enough training, which is better than some literature suggests (~85%). Balanced carry sampling may be compensating.

## Conclusion

Reversed output order is confirmed as a meaningful improvement, especially for training efficiency (2x fewer epochs to 95%). The ~2% accuracy gap at 50 epochs would likely widen with fewer training samples, matching the literature's reported 4-10x sample efficiency advantage.
